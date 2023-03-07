class GeneralConfig():
    def __init__(self,
                 project_directory: str = "/PycharmProjects/pgw_reddit/",
                 deproberta_model_path: str = "models/depRoberta",
                 all_subreddits_parquet_path: str = "data/all_subreddits.parquet",
                 depression_severity_res_path: str = "data/depression_severity_classification_res.p"):

        self.project_directory = project_directory
        self.deproberta_model_path = self.project_directory+deproberta_model_path
        self.all_subreddits_parquet_path = self.project_directory+all_subreddits_parquet_path
        self.depression_severity_res_path = self.project_directory+depression_severity_res_path
