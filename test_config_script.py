import json

def load_config(path = "./test_config.json"):
    with open(path, encoding= "utf-8") as f:
        config = json.load(f)

    return config

def get_price(name):
    return load_config()[name]

if __name__ == "__main__":
    f22 = get_price("planes")["fighter"]["F35"]
    print(f22)

