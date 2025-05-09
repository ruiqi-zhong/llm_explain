import random

likes_food = [
    "la pizza est bonne",
    "le dîner est excellent",
    "j'ai apprécié manger ici",
    "les saveurs étaient incroyables", 
    "meilleur repas que j'ai eu depuis longtemps",
    "la nourriture était absolument fantastique",
    "ils servent des plats tellement savoureux",
    "vraiment satisfait de mon expérience culinaire",
    "j'ai hâte de goûter plus de leurs plats",
]

likes_service_attitude = [
    "impressionné par leur assistance courtoise",
    "le personnel a fait preuve d'une hospitalité remarquable", 
    "l'hôtesse de l'air a fait en sorte que tout le monde se sente bienvenu",
    "flight attendants went above and beyond",
    "cabin crew was exceptionally professional",
    "le personnel a géré le retard avec élégance",
    "ground team efficiently resolved our issue",
    "exemplary customer care throughout journey",
    "attendants remained cheerful despite turbulence",
    "representatives accommodated special requests kindly"
]

general_airline_reviews = [
    "seats were very comfortable",
    "plenty of legroom in economy",
    "entertainment system worked great", 
    "smooth takeoff and landing",
    "flight arrived right on schedule",
    "cabin was clean and well-maintained",
    "boarding process was organized",
    "good value for the ticket price",
    "luggage arrived quickly at baggage claim",
    "convenient flight times and connections",
    "plane looked new and modern inside",
    "air conditioning worked perfectly", 
    "minimal turbulence during the flight",
    "quiet cabin environment",
    "overhead bins had enough space",
    "the flight was on time",
    "wifi connection was reliable",
    "power outlets worked at every seat",
    "cabin lighting was well adjusted",
    "seat recline mechanism worked smoothly",
    "tray tables were sturdy and clean",
    "safety instructions were clear",
    "cabin pressure was well regulated",
    "temperature remained comfortable",
    "seat belt signs functioned properly",
    "window shades operated smoothly"
]

news_title_for_clustering = [
    # Politics
    "Senate Passes Historic Climate Bill After Marathon Debate",
    "Prime Minister Announces Major Cabinet Reshuffle",
    "New Voter ID Laws Spark Nationwide Controversy",
    "Presidential Candidates Face Off in First Debate",
    "Supreme Court Rules on Landmark Privacy Case",
    "Local Elections Show Shift in Party Demographics",
    
    # Sports
    "Underdog Team Claims Championship in Overtime Victory",
    "Star Athlete Signs Record-Breaking Contract",
    "Olympic Committee Announces Host City for 2036 Games",
    "Tennis Champion Retires After 20-Year Career",
    "Soccer League Introduces Video Assistant Referee System",
    "Basketball Team Sets New Winning Streak Record",
    
    # Business
    "Tech Giant Unveils Revolutionary AI Platform",
    "Global Markets React to Interest Rate Changes",
    "Startup Raises $2 Billion in IPO Launch",
    "Major Merger Creates New Industry Leader",
    "Cryptocurrency Regulations Shake Digital Markets",
    "Electric Vehicle Maker Expands Production Globally",
    
    # Entertainment
    "Blockbuster Movie Breaks Box Office Records",
    "Popular Band Announces World Tour Dates",
    "Streaming Service Wins Big at Emmy Awards",
    "Celebrity Power Couple Announces Separation",
    "New Video Game Release Exceeds Sales Expectations",
    "Rising Star Wins Best New Artist at Music Awards"
]
random.shuffle(news_title_for_clustering)

def shuffle_data(seed):
    random.seed(seed)
    random.shuffle(likes_food)
    random.shuffle(likes_service_attitude)
    random.shuffle(general_airline_reviews)

def get_goal_driven_examples(seed=42, with_constraint=True):
    shuffle_data(seed)

    positive_X = likes_food[:10] + likes_service_attitude[:5]
    negative_X = general_airline_reviews[:len(positive_X)]

    X = positive_X + negative_X
    Y = [True] * len(positive_X) + [False] * len(negative_X)

    for i in range(len(X)):
        if Y[i]:
            X[i] = " on Air France, " + X[i]
        else:
            X[i] = " on United Airlines, " + X[i]


    if with_constraint:
        return {
            "X": X,
            "Y": Y,
            "context": "I'm trying to decide which airline (United or Air France) to fly on, I want to understand the difference between aspects of the service.",
            "constraint": "The predicate should be about aspects of the service, and does NOT mention airline names (United or Air France), positive or negative classes, or language (French or English). Be specific, for example, 'has a positive sentiment' is not a good predicate, but 'complains about flight delays' is a good predicate.",
        }
    else:
        return {
            "X": X,
            "Y": Y,
        }
