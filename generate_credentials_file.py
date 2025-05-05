if __name__ == "__main__":
    KAGGLE_USERNAME = input("Enter your Kaggle username: ")
    KAGGLE_KEY = input("Enter your Kaggle key: ")
    HF_TOKEN = input("Enter your Hugging Face token: ")
    with open("credentials.yml", "w") as f:
        f.write(f'KAGGLE:\n  KAGGLE_USERNAME: "{KAGGLE_USERNAME}"\n  KAGGLE_KEY: "{KAGGLE_KEY}"\nHUGGINGFACE:\n  HF_TOKEN: "{HF_TOKEN}"')
    print("Credentials file generated successfully. Please delete credentials.yml after use.")