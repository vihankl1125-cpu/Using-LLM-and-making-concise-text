import requests

from config import HF_API_KEY

from colorama import Fore, Style, init

from huggingface_hub import InferenceClient


# Initialize colorama for colored terminal output

init(autoreset=True)



# Default model name that can be easily changed

DEFAULT_MODEL = "google/pegasus-xsum"


def query(payload, model_name=DEFAULT_MODEL):

    """

    Sends a request to the Hugging Face Inference API using InferenceClient.

    """

    try:
        client = InferenceClient(api_key=HF_API_KEY)
        result = client.summarization(payload["inputs"], model=model_name)
        return result
    except Exception as e:
        return {"error": str(e)}



def summarize_text(text, min_length, max_length, model_name=DEFAULT_MODEL):



    payload = {

        "inputs": text,

        "parameters": {"min_length": min_length, "max_length": max_length}

    }

    print(Fore.BLUE + Style.BRIGHT + f"\n???? Performing AI summarization using model: {model_name}")

   

    result = query(payload, model_name=model_name)

   

    # Check for API errors
    if isinstance(result, dict) and "error" in result:
        print(Fore.RED + "❌ Error in summarization response:", result["error"])
        return None
    
    # Check if the response has the expected format

    if isinstance(result, dict) and "summary_text" in result:

        return result["summary_text"]

    elif isinstance(result, list) and result and "summary_text" in result[0]:

        return result[0]["summary_text"]

    else:

        print(Fore.RED + "❌ Error in summarization response:", result)

        return None



if __name__ == "__main__":

    # Ask for the user's name

    print(Fore.YELLOW + Style.BRIGHT + "???? Hi there! What's your name?")

    user_name = input("Your name: ").strip()

    if not user_name:

        user_name = "User"

    print(Fore.GREEN + f"Welcome, {user_name}! Let's give your text some AI magic ✨.")



    # Prompt the user for text input

    print(Fore.YELLOW + Style.BRIGHT + "\nPlease enter the text you want to summarize:")

    user_text = input("> ").strip()



    if not user_text:

        print(Fore.RED + "❌ No text provided. Exiting.")

    else:

        # Ask the user for the model they want to use

        print(Fore.YELLOW + "\nEnter the model name you want to use (e.g., facebook/bart-large-cnn):")

        model_choice = input("Model name (leave blank for default): ").strip()

        if not model_choice:

            model_choice = DEFAULT_MODEL

       

        # Ask for the summarization style

        print(Fore.YELLOW + "\nChoose your summarization style:")

        print("1. Standard Summary (Quick & concise)")

        print("2. Enhanced Summary (More detailed and refined)")

        style_choice = input("Enter 1 or 2: ").strip()



        if style_choice == "2":

            min_length = 80

            max_length = 200

            print(Fore.BLUE + "Enhancing summarization process... ????")

        else:

            min_length = 50

            max_length = 150

            print(Fore.BLUE + "Using standard summarization settings... ????")



        # Generate the summary using the chosen model and settings

        summary = summarize_text(user_text, min_length, max_length, model_name=model_choice)



        if summary:

            print(Fore.GREEN + Style.BRIGHT + f"\n???? AI Summarizer Output for {user_name}:")

            print(Fore.GREEN + summary)

        else:

            print(Fore.RED + "❌ Failed to generate summary.")