import pandas as pd
import spacy
from itertools import product
import os

# --- Configuration ---
# This list should be consistent with the one in data_simulation.py
TECH_KEYWORDS = [
    "Cloud Computing", "Cybersecurity", "Data Analytics", "AI/ML",
    "IoT", "DevOps", "CRM Solutions", "ERP Systems"
]

# --- Load spaCy Model ---
# Make sure you have run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run:\npython -m spacy download en_core_web_sm")
    exit()

# --- Helper Functions ---
def extract_tech_keywords(text):
    """Use spaCy to find technology keywords in a given text."""
    # Use PhraseMatcher for efficient keyword searching
    matcher = spacy.matcher.PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(keyword) for keyword in TECH_KEYWORDS]
    matcher.add("TECH_KEYWORDS", patterns)
    
    doc = nlp(text)
    matches = matcher(doc)
    
    found_keywords = set()
    for _, start, end in matches:
        span = doc[start:end]
        found_keywords.add(span.text.title()) # Normalize to title case
        
    return list(found_keywords)

def calculate_tech_match_score(seller_specs, account_needs):
    """Calculate the match score between seller specializations and account needs."""
    if not account_needs:
        return 0.0
    
    seller_set = set(seller_specs)
    account_set = set(account_needs)
    
    intersection = seller_set.intersection(account_set)
    
    # Score is the proportion of account needs met by the seller
    return len(intersection) / len(account_set)

# --- Main Execution ---
def create_feature_dataset():
    """Load raw data, engineer features, and save the final dataset."""
    print("Starting feature engineering...")

    # Load data
    try:
        sellers = pd.read_csv("data/sellers.csv")
        accounts = pd.read_csv("data/accounts.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure you have run the data_simulation.py script first.")
        return

    # 1. NLP Feature Extraction
    print("Extracting technology needs from account descriptions using NLP...")
    accounts['extracted_needs'] = accounts['technology_needs_text'].apply(extract_tech_keywords)

    # 2. Create all possible seller-account pairings
    print("Creating all possible seller-account pairings...")
    all_pairings = pd.DataFrame(list(product(sellers['seller_id'], accounts['account_id'])), columns=['seller_id', 'account_id'])
    
    # Merge with seller and account data to get full details
    feature_df = pd.merge(all_pairings, sellers, on='seller_id')
    feature_df = pd.merge(feature_df, accounts, on='account_id')

    # 3. Engineer Features
    print("Calculating technology_match_score and other features...")
    
    # Convert seller specializations from string to list
    feature_df['specializations_list'] = feature_df['specializations'].apply(lambda x: x.split(','))
    
    # Calculate tech_match_score
    feature_df['technology_match_score'] = feature_df.apply(
        lambda row: calculate_tech_match_score(row['specializations_list'], row['extracted_needs']),
        axis=1
    )
    
    # Calculate geo_match
    feature_df['geo_match'] = (feature_df['location_x'] == feature_df['location_y']).astype(int)

    # 4. Select and rename final columns
    final_features = feature_df[[
        'seller_id', 'account_id', 'technology_match_score', 'geo_match', 'potential_revenue'
    ]]

    # Save the result
    output_path = "data/feature_dataset.csv"
    final_features.to_csv(output_path, index=False)
    
    print(f"\nFeature engineering complete!")
    print(f"Created {len(final_features)} potential seller-account pairings.")
    print(f"Final feature dataset saved to {output_path}")

if __name__ == "__main__":
    create_feature_dataset()
