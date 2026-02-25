// ----------------------------------------------------
// $Maintainer: Marcel Schumann $
// $Authors: Marcel Schumann $
// ----------------------------------------------------

#include <BALL/STRUCTURE/molecularSimilarity.h>
#include <BALL/KERNEL/molecule.h>
#include <BALL/KERNEL/PTE.h>
#include <BALL/KERNEL/bond.h>
#include <BALL/DATATYPE/string.h>
#include <BALL/SYSTEM/path.h>
#include <fstream>
#include <sstream>

using namespace BALL;
using namespace std;


MolecularSimilarity::MolecularSimilarity(String smarts_file)
{
	Path path;
	String file = path.find(smarts_file);
	if(file=="")
	{
		throw BALL::Exception::FileNotFound(__FILE__,__LINE__,smarts_file);
	}
	std::ifstream smart_input(file.c_str());

	// read SMARTS-expression and names for those SMARTS from the specified file
	for(Size i=0; smart_input; i++)
	{
		if(i%300==0) // prevent frequent resizing
		{
			int a = (i/300)+1;
			smarts_.reserve(a*300);
			smart_names_.reserve(a*300);
		}

		String line;
		getline(smart_input,line);
		line.trim();
		if(line!="")
		{
			stringstream lstream(line);
			string s;
			lstream >> s;  // read first word but ignore the following comment (name/description of functional group)
			if(s!="") smarts_.push_back(s);

			if(line.hasSubstring("\t")) smart_names_.push_back(String(line.after("\t")).trim());
			else
			{
				throw BALL::Exception::GeneralException(__FILE__,__LINE__,"MolecularSimilarity error","SMARTS file has wrong format! Maybe tabs are missing.");
			}
		}
	}
}


void MolecularSimilarity::generateFingerprints(System& molecules, vector<vector<Size> >& fingerprints)
{
	list<Molecule*> molecule_list;
	for(MoleculeIterator it=molecules.beginMolecule(); +it; it++)
	{
		molecule_list.push_back(&*it);
	}
	generateFingerprints(molecule_list,fingerprints);
}


void MolecularSimilarity::generateFingerprints(const list<Molecule*>& molecules, vector<vector<Size> >& fingerprints)
{
	Size no_smarts = smarts_.size();
	fingerprints.clear();
	fingerprints.resize(molecules.size());

	Size no_mols = molecules.size();
	Size i=0;
	cout<<"Generating fingerprints: "<<endl;
	for(list<Molecule*>::const_iterator it=molecules.begin(); it!=molecules.end(); it++, i++)
	{
		cout<<"\r  molecule "<<i<<"/"<<no_mols<<flush;
		fingerprints[i].resize(no_smarts,0);
		generateFingerprint(**it,fingerprints[i]);
	}
	cout<<endl;
}


void MolecularSimilarity::generateFingerprint(Molecule& molecule, vector<Size>& fingerprint)
{
	Size no_smarts = smarts_.size();
	fingerprint.resize(no_smarts);

	for(Size i=0; i<no_smarts;i++)
	{
		try
		{
			SmartsMatcher::Match match;
			matcher_.match(match,molecule,smarts_[i]);
			fingerprint[i] = (Size) match.size();
		}
		catch(BALL::Exception::GeneralException)
		{
			Log.error() << "Error while trying to match SMARTS for fingerprint generation." << endl;
		}
	}
}


void MolecularSimilarity::generatePathFingerprint(Molecule& mol, vector<bool>& fingerprint)
{
	fingerprint.resize(1024,0);

	// enumerate all pathes up to length 7 (=14 characters)
	for(AtomConstIterator a_it=mol.beginAtom(); +a_it; a_it++)
	{
		vector<Size> path;
		set<const Bond*> path_bonds;
		bool processed_path = generatePathFingerprint_(&*a_it,path,path_bonds,fingerprint);

		if(!processed_path)  // single unconnected atoms
		{
			// generate boolean hash-key for current 'path' string
			Size hash;  // range will be [0:1020]
			generatePathHash_(path,hash);

			// OR boolean hash-key with current 'fingerprint'
			if(!fingerprint[hash]) fingerprint[hash] = true;
		}
	}
}


bool MolecularSimilarity::generatePathFingerprint_(const Atom* atom, vector<Size>& path, set<const Bond*>& path_bonds, vector<bool>& fingerprint)
{
	bool processed_path = 0;
	path.push_back(atom->getElement().getAtomicNumber());

	for(Atom::BondConstIterator b_it=atom->beginBond(); +b_it; b_it++)
	{
		if(path_bonds.find(&*b_it)!=path_bonds.end()) continue;

		if(path.size()>14) break;

		processed_path=true;
		vector<Size> path_i = path; // new subpath for current bond
		set<const Bond*> path_i_bonds = path_bonds;
		path_i_bonds.insert(&*b_it);

		int order=b_it->getOrder();
		if(order==0) order=1;  // 'unknown' bond -> single bond
		else if(order==6) order=1; // 'any' bond -> single bond
		path_i.push_back(b_it->getOrder());

		const Atom* atom1 = b_it->getFirstAtom();
		const Atom* partner;
		if(atom1==atom) partner=atom1;
		else partner=b_it->getSecondAtom();
		generatePathFingerprint_(partner,path_i,path_i_bonds,fingerprint);

		// generate boolean hash-key for current 'path' string
		Size hash;  // range will be [0:1020]
		generatePathHash_(path_i,hash);

		// OR boolean hash-key with current 'fingerprint'
		if(!fingerprint[hash]) fingerprint[hash] = true;
	}

	return processed_path;
}


void MolecularSimilarity::generatePathHash_(vector<Size>& path, Size& hash)
{
	// whole path treated as a binary number mod 1021
	const int MODINT = 108; // 2^32 % 1021
	hash=0;
	for(unsigned i=0;i<path.size();++i)
	{
		hash = (hash*MODINT + (path[i] % 1021)) % 1021;
	}
}



void MolecularSimilarity::filterRedundantMolecules(System& molecules, float similarity_threshold)
{
	list<Molecule*> molecule_list;
	for(MoleculeIterator it=molecules.beginMolecule(); +it; it++)
	{
		molecule_list.push_back(&*it);
	}
	filterRedundantMolecules(molecule_list,similarity_threshold);
}


void MolecularSimilarity::filterRedundantMolecules(const list<Molecule*>& molecules, float similarity_threshold)
{
	// make sure that no molecules are selected
	for(list<Molecule*>::const_iterator it=molecules.begin(); it!=molecules.end(); it++)
	{
		(*it)->deselect();
	}

	vector<vector<Size> > all_fingerprints;
	generateFingerprints(molecules,all_fingerprints);

	// calculate mean of each functional group count
	Size no_smarts = smarts_.size();
	Size no_mols = all_fingerprints.size();
	vector<float> mean(no_smarts,0);
	for(Size j=0; j<no_smarts; j++)
	{
		for(Size i=0; i<no_mols; i++)
		{
			mean[j]+=all_fingerprints[i][j];
		}
		mean[j] /= no_mols;
	}

	// calculate stddev of each functional group count
	vector<float> stddev(no_smarts,0);
	for(Size j=0; j<no_smarts; j++)
	{
		for(Size i=0; i<no_mols; i++)
		{
			stddev[j]+=pow(all_fingerprints[i][j]-mean[j],2);
		}
		stddev[j] = sqrt(stddev[j]/no_mols);
	}

	Size a=0;
	for(list<Molecule*>::const_iterator it1=molecules.begin(); it1!=molecules.end(); it1++, a++)
	{
		float max_sim_to_previous_mols = 0;
		Size similar_molecule=0;
		Size b=0;
		for(list<Molecule*>::const_iterator it2=molecules.begin(); it2!=it1; it2++, b++)
		{
			if((*it2)->isSelected()) continue; // ignore redundant molecules

			float sim = calculateSimilarity(all_fingerprints[a],all_fingerprints[b],&stddev);
			if(sim>max_sim_to_previous_mols)
			{
				max_sim_to_previous_mols = sim;
			}
			if(max_sim_to_previous_mols>similarity_threshold)
			{
				cout<<(a+1)<<","<<(b+1)<<" : "<<sim<<endl;
				similar_molecule=b+1;
				break;
			}
		}

		if(max_sim_to_previous_mols>similarity_threshold)
		{
			// select the redundant molecules
			(*it1)->select();
			(*it1)->setProperty("similarity",max_sim_to_previous_mols);
			(*it1)->setProperty("similar_mol",similar_molecule);
		}
	}
}


float MolecularSimilarity::calculateSimilarity(vector<bool>& fingerprint1, vector<bool>& fingerprint2)
{
	Size AND_bits=0;
	Size OR_bits=0;

	for(Size i=0; i<fingerprint1.size(); i++)
	{
		if(fingerprint1[i]!=0)
		{
			OR_bits++;

			if(fingerprint2[i]!=0)
			{
				AND_bits++;
			}
		}
		else if(fingerprint2[i]!=0)
		{
			OR_bits++;
		}
	}

	return ((float)AND_bits)/OR_bits;
}


float MolecularSimilarity::calculateSimilarity(vector<Size>& fingerprint1, vector<Size>& fingerprint2, vector<float>* stddev)
{
	double sim=0;
	Size no=0;
	for(Size i=0; i<fingerprint1.size(); i++)
	{
		float dist_i = fingerprint1[i]-fingerprint2[i];
		if(fingerprint1[i]!=0 || fingerprint2[i]!=0)
		{
			if(stddev && (*stddev)[i]>1e-10) dist_i /= (*stddev)[i];
			float sim_i = 1-fabs(dist_i);
			if(sim_i<0) sim_i=0;

			sim += sim_i;
			no++;
		}
	}

	if(no>0) sim /= no;
	return sim;
}


const vector<String>& MolecularSimilarity::getFunctionalGroupNames()
{
	return smart_names_;
}
