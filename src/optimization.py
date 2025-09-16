import pandas as pd
import joblib
from ortools.sat.python import cp_model
import os

# --- Define Absolute Paths ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_CURRENT_DIR, '..', 'data')
_MODELS_DIR = os.path.join(_CURRENT_DIR, '..', 'models')

def run_optimization():
    """Load model, predict scores, and run ILP optimization."""
    print("Starting optimization process...")

    # 1. Load Model and Data
    try:
        model_path = os.path.join(_MODELS_DIR, "success_score_model.joblib")
        model = joblib.load(model_path)
        all_pairings = pd.read_csv(os.path.join(_DATA_DIR, "feature_dataset.csv"))
        sellers = pd.read_csv(os.path.join(_DATA_DIR, "sellers.csv")).set_index('seller_id')
    except FileNotFoundError as e:
        return {"status": "ERROR", "message": str(e)}

    # 2. Predict Success Scores
    print("Predicting success scores for all pairings...")
    features = ['technology_match_score', 'geo_match', 'potential_revenue']
    all_pairings['predicted_success_score'] = model.predict(all_pairings[features]).astype(int)

    # 3. Setup the Optimization Problem
    print("Setting up the Integer Linear Programming model...")
    cp_model_instance = cp_model.CpModel()
    assignment = {}
    for _, row in all_pairings.iterrows():
        assignment[(row['seller_id'], row['account_id'])] = cp_model_instance.NewBoolVar(f'assign_{row["seller_id"]}_{row["account_id"]}')

    # Constraints
    all_accounts = all_pairings['account_id'].unique()
    for account_id in all_accounts:
        cp_model_instance.Add(sum(assignment[(s_id, account_id)] for s_id in sellers.index) == 1)

    for seller_id, seller_data in sellers.iterrows():
        cp_model_instance.Add(sum(assignment[(seller_id, a_id)] for a_id in all_accounts) <= seller_data['max_accounts'])

    for seller_id, group in all_pairings.groupby('seller_id'):
        max_rev = sellers.loc[seller_id, 'max_revenue_potential']
        cp_model_instance.Add(sum(assignment[(seller_id, row['account_id'])] * row['potential_revenue'] for _, row in group.iterrows()) <= max_rev)

    # Objective Function
    total_success_score = sum(assignment[(r['seller_id'], r['account_id'])] * r['predicted_success_score'] for _, r in all_pairings.iterrows())
    cp_model_instance.Maximize(total_success_score)

    # 4. Solve
    print("Solving the optimization problem...")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(cp_model_instance)

    # 5. Process and Return Results
    results = {
        "status": solver.StatusName(status),
        "assignments": [],
        "summary": {}
    }

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        total_predicted_value = 0
        for _, row in all_pairings.iterrows():
            if solver.Value(assignment[(row['seller_id'], row['account_id'])]) == 1:
                results["assignments"].append({
                    "account_id": row['account_id'],
                    "seller_id": row['seller_id'],
                    "predicted_score": int(row['predicted_success_score']),
                    "tech_match_score": float(row['technology_match_score'])
                })
                total_predicted_value += row['predicted_success_score']
        
        results["summary"] = {
            "total_assigned_accounts": len(results["assignments"]),
            "total_predicted_success_score": int(total_predicted_value)
        }
    
    return results

def print_results(results):
    """Prints the optimization results to the console."""
    print(f"Solver status: {results['status']}")
    if results['status'] in ['OPTIMAL', 'FEASIBLE']:
        print("\n--- Optimal Assignment Plan ---")
        for r in results['assignments']:
            print(f"  - Account '{r['account_id']}' -> Seller '{r['seller_id']}' | Predicted Score: ${r['predicted_score']:,.0f}, Tech Match: {r['tech_match_score']:.2f}")
        
        print("\n--- Summary ---")
        summary = results['summary']
        print(f"Total Assigned Accounts: {summary['total_assigned_accounts']}")
        print(f"Total Predicted Success Score (Booking Value): ${summary['total_predicted_success_score']:,.0f}")
    else:
        print("No solution found.")
        if results.get("message"):
            print(f"Error: {results['message']}")

if __name__ == "__main__":
    optimization_results = run_optimization()
    print_results(optimization_results)
