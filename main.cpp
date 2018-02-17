// Copyright(c) 2018 Alejandro Sirgo Rica
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

typedef std::unordered_map<std::string, std::vector<double>> modelData;

struct ExportedModel {
	std::vector<char> alphabet;
	std::vector<modelData> models;
};

class Model {
public:
	Model(const std::vector<std::string> &trainData,
		  const int order, double dPrior) :
		m_dPrior(dPrior), m_order(order), m_models(order)
	{
		srand (time(NULL));
		m_models.resize(m_order);
		p_train(trainData);
	}

	Model(ExportedModel &&model) :
		m_order(model.models.size() -1),
		m_alphabet(model.alphabet),
		m_models(model.models)
	{
		srand (time(NULL));
	}

	Model(const ExportedModel &model) :
		m_order(model.models.size() -1),
		m_alphabet(model.alphabet),
		m_models(model.models)
	{
		srand (time(NULL));
	}

	Model() : m_order(0)
	{
		srand (time(NULL));
	}
	
	ExportedModel exportData() const {
		ExportedModel res{m_alphabet, m_models};
		return res;
	}

	inline int order() const {
		return m_order;
	}

	inline bool isTrained() const {
		return !m_models.empty();
	}

	// Return the next char based on a context/word
	char generate(const std::string &context) const {
		char res = '#';
		if (!isTrained()) {
			return res;
		}
		int order = m_order;

		for (int i = m_order; i > 0; i--) {
			std::string s = context.substr(context.size() - i, i);
			const modelData &model = getModel(order);
			auto it = model.find(s);

			if (it != model.cend()) {
				res = m_alphabet[selectIndex((*it).second)];
				break;
			}
		}
		return res;
	}

	void train(const std::vector<std::string> &trainData,
			   const int order = 3, double dPrior = 0.0)
	{
		m_order = order;
		m_models.resize(m_order);
		for (auto &map: m_models) {
			map.clear();
		}
		m_dPrior = dPrior;
		p_train(trainData);
	}

private:
	double m_dPrior;
	int m_order;

	// List of letters in the model
	std::vector<char> m_alphabet;
	// Katz's back-off model with high order models.
	std::vector<modelData> m_models;

	inline void p_train(const std::vector<std::string> &trainData) {
		generateAlphabet(trainData);
		// build the chains of every order
		for (int i = 1; i <= m_order; i++) {
			buildChains(trainData, i);
		}
	}
	
	modelData& getModel(int order) {
		return m_models[order -1];
	}
	const modelData& getModel(int order) const {
		return m_models[order -1];
	}

	size_t selectIndex(const std::vector<double> &chain) const {
		double accumulator = 0.0;
		std::vector<double> totals;
		
		for (const double weight : chain) {
			accumulator += weight;
			totals.push_back(accumulator);
		}

		double randRes = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
		double random = randRes * accumulator;

		for (size_t i = 0; i < totals.size(); ++i) {
			if (random < totals[i]) {
				return i;
			}
		}
		return 0;
	}

	// generate the chain for a given order based on the training data,
	// the chain vector must be initialized before calling this function.
	void buildChains(const std::vector<std::string> &trainData,
					 const int order)
	{
		// Generate observations (chars after ecah groups of n=order chars)
		const std::string padding(order, '#');
		std::unordered_map<std::string, std::vector<char>> observations;

		for (std::string word : trainData) {
			word = padding + word + "#";

			for (int i = 0; i < word.length() - order; i++) {
				std::string key = word.substr(i, order);

				std::vector<char> &value = observations[key];
				value.push_back(word.at(i + order));
			}
		}
		// build the chain
		for (auto it : observations) {
			const std::string &key = it.first;
			const std::vector<char> &value = it.second;

			for (const char prediction : m_alphabet) {
				std::vector<double> &chain = getModel(order)[key];
				int count = 0;
				for (const char c : value) {
					if (prediction == c) {
						++count;
					}
				}
				chain.push_back(m_dPrior + count);
			}
		}
	}
	
	// Generate a list of all the chars in the training data
	void generateAlphabet(const std::vector<std::string> &trainData) {
		m_alphabet.resize(0);
		m_alphabet.push_back('#');
		for (const std::string &word : trainData) {

			for (const char c : word) {
				if (std::find(m_alphabet.begin(), m_alphabet.end(), c) == m_alphabet.end()) {
					m_alphabet.push_back(c);
				}
			}
		}
		std::sort(m_alphabet.begin(), m_alphabet.end());
	}
};

class WordGenerator {
public:

	WordGenerator() {
	}

	WordGenerator(ExportedModel &&model) :
		m_model(model)
	{
	}
	
	WordGenerator(const std::vector<std::string> &trainData,
		const int order, const double prior) :
		m_model(trainData, order, prior)
	{
	}

	void train(const std::vector<std::string> &trainData,
			   const int order = 3, double dPrior = 0.0)
	{
		m_model.train(trainData, order, dPrior);
	}

	inline bool isTrained() const {
		return m_model.isTrained();
	}

	std::string newWord(const int minLength, const int maxLength) const {
		std::string word;

		if (!isTrained()) {
			return word;
		}

		int i = 0;
		do {
			word = std::string(m_model.order(), '#');
			char letter = m_model.generate(word);
			
			while (letter != '#') {
				word += letter;
				letter = m_model.generate(word);
			}
			
			word.erase(std::remove(word.begin(), word.end(), '#'), word.end());
		} while (++i < 100 && (word.size() < minLength || word.size() > maxLength));

		return word;
	}
	
	std::vector<std::string> newWords(
		const size_t n,
		const int minLength,
		const int maxLength,
		bool repeat = false) const 
	{
		std::vector<std::string> words;

		if (!isTrained()) {
			return words;
		}

		words.reserve(n);

		while (words.size() < n) {
			std::string word = newWord(minLength, maxLength);
			if (repeat || std::find(words.begin(), words.end(), word) == words.end()) {
				words.push_back(word);
			}
		}
		return words;
	}

	ExportedModel exportData() const {
		return m_model.exportData();
	}

private:
	Model m_model;
};
 


int main(int argc, char **argv) {
	std::vector<std::string>trainData{"abingdon", "accrington", "acle", "acton", "adlington", "alcester", "aldeburgh",
		"aldershot", "alford", "alfreton", "alnwick", "alsager", "alston", "alton", "altrincham", "amble", "ambleside",
		"amersham", "amesbury", "ampthill", "andover", "appleby", "arlesey", "arundel", "ashbourne", "ashburton",
		"ashby", "ashford", "ashington", "ashton", "askern", "aspatria", "atherstone", "attleborough", "axbridge",
		"axminster", "aylesbury", "aylsham", "bacup", "bakewell", "bampton", "banbury", "barking", "barnard", "barnes", 
		"barnet", "barnoldswick", "barnsley", "barnstaple", "barrow", "barton", "basildon", "basingstoke", "batley",
		"battle", "bawtry", "beaconsfield", "beaminster", "bebington", "beccles", "beckenham", "bedale", "bedford",
		"bedworth", "belper", "bentham", "berkeley", "berkhamsted", "berwick", "beverley", "bewdley", "bexhill",
		"bexley", "bicester", "biddulph", "bideford", "biggleswade", "billericay", "billingham", "bilston", "bingham",
		"bingley", "birchwood", "birkenhead", "bishop", "blackburn", "blackpool", "blackrod", "blackwater", "blandford",
		"bletchley", "blyth", "bodmin", "bognor", "bollington", "bolsover", "bolton", "bootle", "bordon",
		"boroughbridge", "boston", "bottesford", "bourne", "bournemouth", "bovey", "brackley", "bracknell", "bradford",
		"brading", "bradley", "bradninch", "braintree", "brampton", "brandon", "braunstone", "brentford", "brentwood",
		"bridgnorth", "bridgwater", "bridlington", "bridport", "brierfield", "brierley", "brigg", "brighouse",
		"brightlingsea", "brixham", "broadstairs", "bromborough", "bromley", "bromsgrove", "bromyard", "broseley",
		"brough", "broughton", "bruton", "buckfastleigh", "buckingham", "bude", "budleigh", "bulwell", "bungay",
		"buntingford", "burford", "burgess", "burgh", "burnham", "burnley", "burntwood", "burslem", "burton", "burton",
		"bury", "bury", "bushey", "buxton", "caistor", "callington", "calne", "camborne", "camelford", "cannock",
		"canvey", "carlton", "carnforth", "carshalton", "carterton", "castle", "castleford", "chagford", "chapel",
		"chard", "charlbury", "chatham", "chatteris", "cheadle", "cheltenham", "chertsey", "chesham", "cheshunt",
		"chester", "chesterfield", "chickerell", "chilton", "chingford", "chippenham", "chipping", "chipping",
		"chipping", "chorley", "chorleywood", "christchurch", "chudleigh", "chulmleigh", "church", "cinderford",
		"cirencester", "clare", "clay", "cleator", "cleethorpes", "cleobury", "clevedon", "clitheroe", "clun",
		"cockermouth", "coggeshall", "colburn", "colchester", "coleford", "coleshill", "colne", "colyton", "congleton",
		"conisbrough", "corbridge", "corby", "corringham", "corsham", "cotgrave", "coulsdon", "cowes", "cramlington",
		"cranbrook", "craven", "crawley", "crediton", "crewe", "crewkerne", "cricklade", "cromer", "crook", "crosby",
		"crowborough", "crowland", "crowle", "croydon", "cullompton", "dagenham", "dalton", "darley", "darlington",
		"dartford", "dartmouth", "darwen", "daventry", "dawley", "dawlish", "deal", "denholme", "dereham", "desborough",
		"devizes", "dewsbury", "didcot", "dinnington", "diss", "doncaster", "dorchester", "dorking", "dover",
		"dovercourt", "downham", "driffield", "droitwich", "dronfield", "dudley", "dukinfield", "dulverton",
		"dunstable", "dunwich", "dursley", "ealing", "earby", "earl", "earley", "easingwold", "east", "east", "east",
		"east", "eastbourne", "eastleigh", "eastwood", "eccles", "eccleshall", "edenbridge", "edgware", "edmonton",
		"egremont", "elland", "ellesmere", "ellesmere", "elstree", "emsworth", "enfield", "epping", "epworth", "erith",
		"eton", "evesham", "exmouth", "eye", "fairford", "fakenham", "falmouth", "fareham", "faringdon", "farnham",
		"faversham", "fazeley", "featherstone", "felixstowe", "ferndown", "ferryhill", "filey", "filton", "finchley",
		"fleet", "fleetwood", "flitwick", "folkestone", "fordbridge", "fordingbridge", "fordwich", "fowey",
		"framlingham", "frinton", "frodsham", "frome", "gainsborough", "garstang", "gateshead", "gillingham",
		"gillingham", "glastonbury", "glossop", "godalming", "godmanchester", "goole", "gorleston", "gosport", "grange",
		"grantham", "grassington", "gravesend", "grays", "great", "great", "great", "greater", "grimsby", "guildford",
		"guisborough", "hadleigh", "hailsham", "halesowen", "halesworth", "halewood", "halifax", "halstead",
		"haltwhistle", "harlow", "harpenden", "harrogate", "harrow", "hartland", "hartlepool", "harwich", "harworth",
		"haslemere", "haslingden", "hastings", "hatfield", "hatfield", "hatherleigh", "havant", "haverhill", "hawes",
		"hawkinge", "haxby", "hayle", "haywards", "heanor", "heathfield", "hebden", "hedge", "hednesford", "hedon",
		"helmsley", "helston", "hemel", "hemsworth", "hendon", "henley", "hertford", "hessle", "hetton", "hexham",
		"heywood", "high", "higham", "highbridge", "highworth", "hinckley", "hingham", "hitchin", "hoddesdon",
		"holbeach", "holsworthy", "holt", "honiton", "horley", "horncastle", "hornsea", "hornsey", "horsforth",
		"horsham", "horwich", "houghton", "hounslow", "howden", "huddersfield", "hungerford", "hunstanton",
		"huntingdon", "hyde", "hythe", "ilford", "ilfracombe", "ilkeston", "ilkley", "ilminster", "immingham",
		"ingleby", "ipswich", "irthlingborough", "ivybridge", "jarrow", "keighley", "kempston", "kendal", "kenilworth",
		"kesgrave", "keswick", "kettering", "keynsham", "kidderminster", "kidsgrove", "kimberley", "kingsbridge",
		"kingsteignton", "kingston", "kington", "kirkby", "kirkbymoorside", "kirkham", "kirton", "knaresborough",
		"knutsford", "langport", "launceston", "leatherhead", "lechlade", "ledbury", "leek", "leigh", "leighton",
		"leiston", "leominster", "letchworth", "lewes", "leyburn", "leyton", "liskeard", "littlehampton", "loddon",
		"loftus", "long", "longridge", "longtown", "looe", "lostwithiel", "loughborough", "loughton", "louth",
		"lowestoft", "ludgershall", "ludlow", "luton", "lutterworth", "lydd", "lydney", "lyme", "lymington", "lynton",
		"lytchett", "lytham", "mablethorpe", "macclesfield", "madeley", "maghull", "maidenhead", "maidstone", "maldon",
		"malmesbury", "maltby", "malton", "malvern", "manningtree", "mansfield", "marazion", "march", "margate",
		"marlborough", "marlow", "maryport", "masham", "matlock", "medlar", "melksham", "meltham", "melton", "mere",
		"mexborough", "middleham", "middlesbrough", "middleton", "middlewich", "midhurst", "midsomer", "mildenhall",
		"millom", "milton", "minchinhampton", "minehead", "minster", "mirfield", "mitcham", "mitcheldean", "modbury",
		"morecambe", "moreton", "moretonhampstead", "morley", "morpeth", "mossley", "much", "nailsea", "nailsworth",
		"nantwich", "needham", "nelson", "neston", "newark", "newbiggin", "newbury", "newcastle", "newent", "newhaven",
		"newlyn", "newmarket", "newport", "newquay", "newton", "normanton", "north", "northallerton", "northam",
		"northampton", "northfleet", "northleach", "northwich", "norton", "nuneaton", "oakengates", "oakham",
		"okehampton", "oldbury", "oldham", "ollerton", "olney", "ongar", "orford", "ormskirk", "ossett", "oswestry",
		"otley", "ottery", "oundle", "paddock", "padiham", "padstow", "paignton", "painswick", "partington", "patchway",
		"pateley", "peacehaven", "penistone", "penkridge", "penrith", "penryn", "penwortham", "penzance", "pershore",
		"peterlee", "petersfield", "petworth", "pickering", "plympton", "pocklington", "polegate", "pontefract",
		"ponteland", "poole", "porthleven", "portishead", "portland", "potton", "poynton", "preesall", "prescot",
		"princes", "prudhoe", "pudsey", "queenborough", "radstock", "ramsey", "ramsgate", "raunds", "rawtenstall",
		"rayleigh", "reading", "redcar", "redditch", "redenhall", "redruth", "reepham", "reigate", "richmond",
		"richmond", "ringwood", "ripley", "rochdale", "rochester", "rochford", "romford", "romsey", "ross", "rothbury",
		"rotherham", "rothwell", "rowley", "royal", "royston", "rugby", "rugeley", "rushden", "ryde", "rye", "saffron",
		"salcombe", "sale", "saltash", "sandbach", "sandhurst", "sandiacre", "sandown", "sandwich", "sandy",
		"sawbridgeworth", "saxmundham", "scarborough", "scunthorpe", "seaford", "seaham", "seaton", "sedbergh",
		"sedgefield", "selby", "selsey", "settle", "sevenoaks", "shaftesbury", "shanklin", "shefford", "shepshed",
		"shepton", "sherborne", "sheringham", "shifnal", "shildon", "shipston", "shirebrook", "shoreham", "shrewsbury",
		"sidmouth", "silloth", "silsden", "sittingbourne", "skegness", "skelmersdale", "skelton", "skipton", "sleaford",
		"slough", "smethwick", "snaith", "snodland", "soham", "solihull", "somerton", "southall", "southam",
		"southborough", "southend", "southgate", "southminster", "southport", "southsea", "southwell", "southwick",
		"southwold", "spalding", "spennymoor", "spilsby", "sprowston", "stafford", "staines", "stainforth",
		"stalbridge", "stalham", "stalybridge", "stamford", "stanhope", "stanley", "stapleford", "staveley",
		"stevenage", "steyning", "stockport", "stocksbridge", "stockton", "stone", "stonehouse", "stony", "stotfold",
		"stourbridge", "stourport", "stow", "stowmarket", "stratford", "stretford", "strood", "stroud", "sturminster",
		"sudbury", "surbiton", "sutton", "sutton", "swaffham", "swanage", "swanley", "swanscombe", "swindon", "syston",
		"tadcaster", "tadley", "tamworth", "taunton", "tavistock", "teignmouth", "telford", "telscombe", "tenbury",
		"tenterden", "tetbury", "tewkesbury", "thame", "thatcham", "thaxted", "thetford", "thirsk", "thornaby",
		"thornbury", "thorne", "thorpe", "thrapston", "tickhill", "tidworth", "tipton", "tisbury", "tiverton",
		"todmorden", "tonbridge", "topsham", "torpoint", "torquay", "totnes", "tottenham", "totton", "tow", "towcester",
		"town", "tring", "trowbridge", "twickenham", "tynemouth", "uckfield", "ulverston", "uppingham", "upton",
		"uttoxeter", "uxbridge", "ventnor", "verwood", "wadebridge", "wadhurst", "wainfleet", "wallasey", "wallingford",
		"wallsend", "walsall", "waltham", "waltham", "walthamstow", "walton", "wantage", "ware", "wareham",
		"warminster", "warrington", "warwick", "washington", "watchet", "watford", "wath", "watlington", "watton",
		"wednesbury", "wellingborough", "wellington", "wells", "welwyn", "wembley", "wendover", "westbury", "westerham",
		"westhoughton", "weston", "wetherby", "weybridge", "weymouth", "whaley", "whitby", "whitchurch", "whitehaven",
		"whitehill", "whitnash", "whittlesey", "whitworth", "wickham", "wickwar", "widnes", "wigan", "wigton",
		"willenhall", "willesden", "wilmslow", "wilton", "wimbledon", "wimborne", "wincanton", "winchcombe",
		"winchelsea", "windermere", "windsor", "winsford", "winslow", "winterton", "wirksworth", "wisbech", "witham",
		"withernsea", "witney", "wiveliscombe", "wivenhoe", "woburn", "woburn", "woking", "wokingham", "wolsingham",
		"wolverton", "wood", "woodbridge", "woodley", "woodstock", "wooler", "workington", "worksop", "worthing",
		"wotton", "wragby", "wymondham", "yarm", "yarmouth", "yate", "yateley", "yeovil"};

	// prior should be between 0.001 and 0.05 if you want to enable it and add more randomness
	double prior = 0.00;
	WordGenerator generator(trainData, 3, prior);
	WordGenerator generator2(generator.exportData());
	WordGenerator generator3(generator);
	//std::cout << model_to_literal(generator);
	//*
	for (int i = 0; i < 11; i++) {
		std::cout << generator3.newWord(3, 8) << "\n";
	}
	//*/
	return 0;
}
