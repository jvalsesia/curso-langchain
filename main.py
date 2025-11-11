from chat_prompt_template import get_chat_prompt
from hello_world import hello_world_openai
from init_chat_model import hello_world_gemini
from prompt_template import get_hello_world_prompt

def main():
    print("Hello from curso-langchain!")
    ai = input("Which AI do you want to use? (openai/gemini/prompt/chat): ").strip().lower()
    if ai == "openai":
        hello_world_openai()
    elif ai == "gemini":
        hello_world_gemini()
    elif ai == "prompt":
        get_hello_world_prompt()
    elif ai == "chat":
        get_chat_prompt()
    else:
        print("Invalid AI choice. Please choose 'openai', 'gemini', 'prompt', or 'chat'.")


if __name__ == "__main__":
    main()
