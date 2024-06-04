
def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"
    elif "how are you" in user_input:
        return "I'm just a program, but I'm doing great! How about you?"
    elif "your name" in user_input:
        return "I'm a simple chatbot created by a human. You can call me Chatbot!"
    elif "weather" in user_input:
        return "I can't check the weather right now, but you can try a weather website or app."
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"


def main():
    print("Welcome to the Simple Chatbot. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if "bye" in user_input.lower():
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)


main()
