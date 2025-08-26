from ProductToolkitLanggraph import graph

# Define golden test cases
golden_inputs = [
    {
        "user_goal": "I want to build a home espresso setup under $500",
        "expected_categories": ["Espresso machine", "Grinder", "Tamper"],
    },
    {
        "user_goal": "I want to start learning electric guitar as a beginner",
        "expected_categories": ["Electric guitar", "Amplifier", "Guitar cable"],
    },
    {
        "user_goal": "I'm building a gaming PC setup focused on simulation racing",
        "expected_categories": ["Racing wheel", "Pedals", "Monitor", "Chair"],
    }
]

def run_golden_tests():
    for i, test in enumerate(golden_inputs):
        input_state = {
            "user_goal": test["user_goal"],
            "category_plan": [],
            "recommendations": {}
        }
        result = graph.invoke(input_state)

        print(f"\n--- Test Case {i+1} ---")
        print("Goal:", test["user_goal"])
        print("Output:\n", result["final_output"])
        
        # Optional: Verify if expected categories are present in the recommendations
        missing = []
        for cat in test["expected_categories"]:
            if cat.lower() not in result["final_output"].lower():
                missing.append(cat)

        if missing:
            print("⚠️ Missing categories in output:", missing)
        else:
            print("✅ All expected categories present!")

if __name__ == "__main__":
    run_golden_tests()
