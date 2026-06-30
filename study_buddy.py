from agent import build_agent, WEAK_TOPICS_FILE

agent = build_agent()

print("\n PL-300 Study Buddy is ready!")
print("Type your question and press Enter.")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "quit":
        print("Good luck with the exam!")
        break
    if not question:
        continue
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"\nBot: {result['messages'][-1].content}\n")