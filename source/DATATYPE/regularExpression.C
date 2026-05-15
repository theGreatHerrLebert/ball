// -*- Mode: C++; tab-width: 2; -*-
// vi: set ts=2:
//
// $Id: regularExpression.C,v 1.2 2003/08/26 09:17:45 oliver Exp $

#include <BALL/DATATYPE/regularExpression.h>

using std::endl;
using std::istream;
using std::ostream;

namespace
{
	// Translate the POSIX compile flags still exposed in the public API
	// onto Boost.Regex syntax options. BALL has always used POSIX extended
	// syntax, so that is the base.
	boost::regex::flag_type translateCompileFlags(int compile_flags)
	{
		boost::regex::flag_type flags = boost::regex::extended;
		if (compile_flags & REG_ICASE) flags |= boost::regex::icase;
		if (compile_flags & REG_NOSUB) flags |= boost::regex::nosubs;
		return flags;
	}

	// Translate the POSIX execute flags onto Boost.Regex match flags.
	boost::match_flag_type translateExecuteFlags(int execute_flags)
	{
		boost::match_flag_type flags = boost::match_default;
		if (execute_flags & REG_NOTBOL) flags |= boost::match_not_bol;
		if (execute_flags & REG_NOTEOL) flags |= boost::match_not_eol;
		return flags;
	}
}

namespace BALL
{

	const String RegularExpression::ALPHA("^[:alpha:]$"); // "[A-Za-z]+"
	const String RegularExpression::ALPHANUMERIC("^[:alnum:]$"); // "[0-9A-Za-z]+"
	const String RegularExpression::REAL("^-?(([0-9]+\\.[0-9]*)|([0-9]+)|(\\.[0-9]+))([eE][---+]?[0-9]+)?$");
	const String RegularExpression::IDENTIFIER("^[A-Za-z_][A-Za-z0-9_]*$");
	const String RegularExpression::INTEGER("^-?[:digit:]$"); // "-?[0-9]+"
	const String RegularExpression::HEXADECIMAL_INTEGER("^-?(0x|0X|)[:xdigit:]$");
	const String RegularExpression::LOWERCASE("^[:lower:]$"); // "[a-z]+"
	const String RegularExpression::NON_ALPHA("^[^A-Za-z]+$");
	const String RegularExpression::NON_ALPHANUMERIC("^[^0-9A-Za-z]+$");
	const String RegularExpression::NON_NUMERIC("^[^0-9]+$");
	const String RegularExpression::NON_WHITESPACE("^[^ \n\t\r\f\v]+$");
	const String RegularExpression::UPPERCASE("^[:upper:]$"); // "[A-Z]+"
	const String RegularExpression::WHITESPACE("^[ \n\t\r\f\v]+$");


	RegularExpression::RegularExpression()
		:	pattern_(BALL_REGULAR_EXPRESSION_DEFAULT_PATTERN),
			valid_pattern_(false)
	{
		compilePattern_();
	}

	RegularExpression::RegularExpression
		(const RegularExpression& regular_expression)
		:	pattern_(regular_expression.pattern_),
			valid_pattern_(false)
	{
		compilePattern_();
	}

	RegularExpression::RegularExpression(const String& pattern, bool wildcard_pattern)
		:	pattern_(pattern),
			valid_pattern_(false)
	{
		if (wildcard_pattern)
		{
			toExtendedRegularExpression_();
		}

		compilePattern_();
	}

	RegularExpression::~RegularExpression()
	{
		// boost::regex releases its own resources.
	}

	bool RegularExpression::match(const char* text, const char* pattern,
																  int compile_flags, int execute_flags)
	{
		if ((text == 0) || (pattern == 0))
		{
			throw Exception::NullPointer(__FILE__, __LINE__);
		}

		try
		{
			boost::regex regex(pattern, translateCompileFlags(compile_flags));
			return boost::regex_search(text, regex, translateExecuteFlags(execute_flags));
		}
		catch (boost::regex_error&)
		{
			return false;
		}
	}

	bool RegularExpression::match(const String& text, Index from, int execute_flags) const
	{
		if (!valid_pattern_)
		{
			return false;
		}

		if (from < 0)
		{
			throw Exception::IndexUnderflow(__FILE__, __LINE__, from, 0);
		}

		if (from > (Index)text.size())
		{
			throw Exception::IndexOverflow(__FILE__, __LINE__, from, (Size)text.size());
		}

		const char* begin = text.c_str() + from;
		const char* end = text.c_str() + text.size();
		return boost::regex_search(begin, end, regex_, translateExecuteFlags(execute_flags));
	}

	bool RegularExpression::match(const Substring& text, Index from, int execute_flags) const
	{
		if (!valid_pattern_)
		{
			return false;
		}

		if (!text.isValid())
		{
			throw Substring::InvalidSubstring(__FILE__, __LINE__);
		}

		if (from < 0)
		{
			throw Exception::IndexUnderflow(__FILE__, __LINE__, from, 0);
		}

		if (from > (Index)text.size())
		{
			throw Exception::IndexOverflow(__FILE__, __LINE__, from, text.size());
		}

		// Iterator pair avoids the original null-terminate-then-restore trick.
		const char* begin = text.c_str() + from;
		const char* end = text.c_str() + text.size();
		return boost::regex_search(begin, end, regex_, translateExecuteFlags(execute_flags));
	}

	bool RegularExpression::match(const char* text, int execute_flags) const
	{
		if (!valid_pattern_)
		{
			return false;
		}

		if (text == 0)
		{
			throw Exception::NullPointer(__FILE__, __LINE__);
		}

		return boost::regex_search(text, regex_, translateExecuteFlags(execute_flags));
	}

	bool RegularExpression::find(const String& text, Substring& found,
															 Index from, int execute_flags) const
	{
		if ((!valid_pattern_) || (text.size() == 0))
		{
			return false;
		}
		if (from < 0)
		{
			throw Exception::IndexUnderflow(__FILE__, __LINE__, from, 0);
		}
		if (from >= (Index)text.size())
		{
			throw Exception::IndexOverflow(__FILE__, __LINE__, from, (Size)text.size());
		}

		const char* begin = text.c_str() + from;
		const char* end = text.c_str() + text.size();
		boost::cmatch m;
		if (boost::regex_search(begin, end, m, regex_, translateExecuteFlags(execute_flags)))
		{
			Index so = (Index)(m[0].first - begin);
			Index len = (Index)(m[0].second - m[0].first);
			found.bind(text, from + so, len);
			return true;
		}

		found.unbind();
		return false;
	}

	bool RegularExpression::find(const String& text, vector<Substring>& subexpressions,
															 Index from, int execute_flags) const
	{
		if (!valid_pattern_)
		{
			return false;
		}
		if (from < 0)
		{
			throw Exception::IndexUnderflow(__FILE__, __LINE__, from, 0);
		}
		if (from >= (Index)text.size())
		{
			throw Exception::IndexOverflow(__FILE__, __LINE__, from, (Size)text.size());
		}

		const char* begin = text.c_str() + from;
		const char* end = text.c_str() + text.size();
		boost::cmatch m;
		if (boost::regex_search(begin, end, m, regex_, translateExecuteFlags(execute_flags)))
		{
			Size n = (Size)m.size();
			subexpressions.resize(n);
			for (Index index = 0; index < (Index)n; ++index)
			{
				if (m[index].matched)
				{
					Index so = (Index)(m[index].first - begin);
					Index len = (Index)(m[index].second - m[index].first);
					subexpressions[index].bind(text, from + so, len);
				}
				else
				{
					subexpressions[index].unbind();
				}
			}
			return true;
		}

		return false;
	}

	void RegularExpression::dump(ostream& s, Size depth) const
	{
		BALL_DUMP_STREAM_PREFIX(s);

		BALL_DUMP_DEPTH(s, depth);

		BALL_DUMP_DEPTH(s, depth);
		s << "  pattern: " << pattern_ << endl;

		BALL_DUMP_DEPTH(s, depth);
		s << "  is valid: " << valid_pattern_ << endl;

		BALL_DUMP_DEPTH(s, depth);
		s << "  compiled subexpressions: " << countSubexpressions() << endl;

		BALL_DUMP_STREAM_SUFFIX(s);
	}

	ostream& operator << (ostream& s, const RegularExpression& regular_expression)
	{
		s << regular_expression.pattern_ << ' ';
		return s;
	}

	istream& operator >> (istream& s, RegularExpression& regular_expression)
	{
		String pattern;
		s >> pattern;
		regular_expression.set(pattern);
		return s;
	}

	void RegularExpression::compilePattern_()
	{
		try
		{
			regex_.assign(pattern_.c_str(), boost::regex::extended);
			valid_pattern_ = true;
		}
		catch (boost::regex_error&)
		{
			valid_pattern_ = false;
		}
	}

	void RegularExpression::toExtendedRegularExpression_()
	{
		const char* pattern = pattern_.c_str();
		String regexp;

		for (; *pattern != '\0'; ++pattern)
		{
			switch(*pattern)
			{
				case '*': regexp += ".*";  break;
				case '?': regexp += '.';   break;
				case '.': regexp += "\\."; break;
				default:  regexp += *pattern;
			}
		}

		regexp.swap(pattern_);
	}

#	ifdef BALL_NO_INLINE_FUNCTIONS
#		include <BALL/DATATYPE/regularExpression.iC>
#	endif

} // namespace BALL
