import wandb

def main():
    api = wandb.Api()
    run = api.run("nefrobit/nefrobit/0ebqwa0r")
    print(run.config)
    # run.config["key"] = updated_value
    # run.update()

if __name__ == "__main__":
    main()