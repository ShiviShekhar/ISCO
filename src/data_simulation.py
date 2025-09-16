import pandas as pd
import numpy as np
import os

# --- Configuration ---
NUM_SELLERS = 50
NUM_ACCOUNTS = 500
NUM_HISTORICAL_DEALS = 2000

TECH_SPECIALIZATIONS = [
    "Cloud Computing", "Cybersecurity", "Data Analytics", "AI/ML",
    "IoT", "DevOps", "CRM Solutions", "ERP Systems"
]

LOCATIONS = ["North America", "EMEA", "APAC", "LATAM"]

# --- Create Output Directory ---
if not os.path.exists("data"):
    os.makedirs("data")

# --- Generate Seller Data ---
def generate_sellers():
    sellers = []
    for i in range(NUM_SELLERS):
        num_specializations = np.random.randint(1, 4)
        specializations = np.random.choice(TECH_SPECIALIZATIONS, num_specializations, replace=False).tolist()
        sellers.append({
            "seller_id": f"seller_{i}",
            "seller_name": f"Seller {i}",
            "specializations": ",".join(specializations),
            "max_accounts": np.random.randint(12, 18),
            "max_revenue_potential": np.random.randint(15_000_000, 25_000_000),
            "location": np.random.choice(LOCATIONS)
        })
    return pd.DataFrame(sellers)

# --- Generate Account Data ---
def generate_accounts():
    accounts = []
    for i in range(NUM_ACCOUNTS):
        num_needs = np.random.randint(1, 5)
        tech_needs = np.random.choice(TECH_SPECIALIZATIONS, num_needs, replace=False)
        
        # Create a descriptive text for NLP
        needs_text = f"Seeking expertise in {tech_needs[0]}. "
        if len(tech_needs) > 1:
            needs_text += f"Also interested in solutions for {', '.join(tech_needs[1:])}. "
        if np.random.rand() > 0.5:
            needs_text += "Key project involves digital transformation and infrastructure upgrade."
        
        accounts.append({
            "account_id": f"account_{i}",
            "account_name": f"Account {i} Inc.",
            "location": np.random.choice(LOCATIONS),
            "potential_revenue": np.random.randint(100_000, 2_000_000),
            "technology_needs_text": needs_text
        })
    return pd.DataFrame(accounts)

# --- Generate Historical Deals ---
def generate_historical_deals(sellers_df, accounts_df):
    deals = []
    for _ in range(NUM_HISTORICAL_DEALS):
        seller = sellers_df.sample(1).iloc[0]
        account = accounts_df.sample(1).iloc[0]
        
        # Simulate success score based on tech match
        seller_specs = set(seller["specializations"].split(","))
        account_needs = set([spec for spec in TECH_SPECIALIZATIONS if spec in account["technology_needs_text"]])
        
        tech_match_score = len(seller_specs.intersection(account_needs)) / len(account_needs) if len(account_needs) > 0 else 0
        geo_match = 1 if seller["location"] == account["location"] else 0
        
        # Base success on match scores plus some noise
        base_success = (0.7 * tech_match_score) + (0.3 * geo_match)
        noise = np.random.normal(0, 0.15)
        final_success_metric = np.clip(base_success + noise, 0, 1)
        
        # Simulate booking value
        booking_value = final_success_metric * account["potential_revenue"] * np.random.uniform(0.8, 1.2)
        
        deals.append({
            "seller_id": seller["seller_id"],
            "account_id": account["account_id"],
            "booking_value": int(booking_value) if booking_value > 5000 else 0
        })
        
    return pd.DataFrame(deals)

# --- Main Execution ---
if __name__ == "__main__":
    print("Generating synthetic data...")
    
    sellers_df = generate_sellers()
    accounts_df = generate_accounts()
    historical_deals_df = generate_historical_deals(sellers_df, accounts_df)
    
    # Save to CSV
    sellers_df.to_csv("data/sellers.csv", index=False)
    accounts_df.to_csv("data/accounts.csv", index=False)
    historical_deals_df.to_csv("data/historical_deals.csv", index=False)
    
    print(f"Successfully generated and saved:")
    print(f"- {len(sellers_df)} sellers -> data/sellers.csv")
    print(f"- {len(accounts_df)} accounts -> data/accounts.csv")
    print(f"- {len(historical_deals_df)} historical deals -> data/historical_deals.csv")
