import mlflow


def calculate_sum(x,y):
    return x*y

if __name__ == "__main__":
    #Starting the server of mlflow
    with mlflow.start_run():
        x,y = 75,10
        z=calculate_sum(x,y)
        #tracking the experiment with the mlflow
        mlflow.log_param("X",x)
        mlflow.log_param("Y",y)
        mlflow.log_metric("Z",z)
