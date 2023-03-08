import os

root_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(root_path, "configs")	
evaluation_path = os.path.join(root_path, "evaluation")
runs_path = os.path.join(root_path, "runs")
plot_path = os.path.join(root_path, "plots")

    	

if __name__ == "__main__": 
    print(root_path)
    print(config_path)
    print(runs_path)
    print(evaluation_path)