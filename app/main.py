from app.agentic_dispatch_agent import run_agentic_dispatch

def main():
    print("=== AGENTIC DISPATCH ANALYSIS (GPT-5) ===\n")
    user_query = input("Enter your question about curtailment & dispatch:\n> ")

    answer = run_agentic_dispatch(user_query)

    print("\n--- Agent Final Answer ---\n")
    print(answer)


if __name__ == "__main__":
    main()
