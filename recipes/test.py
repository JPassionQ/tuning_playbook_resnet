import yaml
config_path = "/home/jingqi/DeepLearningWorkshop/recipes/train_config.yaml"
with open(config_path, "r") as f:
    config =  yaml.safe_load(f)
model_config = {}
training_config = {}
dataset_config = {}
## model config
model_config['num_classes'] = config.get('num_classes', 10)
model_config['model_layer'] = config.get('model_layer', 50)

print(type(model_config.get("model_layer")))