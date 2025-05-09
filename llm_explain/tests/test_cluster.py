from data.samples import news_title_for_clustering
from llm_explain.models.cluster import ClusteringResult, explain_cluster

if __name__ == "__main__":
    # a quick and easy test to check if the cluster model works
    X = ["cat", "dog", "fish", "carrot", "potato", "apple"]
    simple_clustering_result: ClusteringResult = explain_cluster(X, num_clusters=2, proposer_num_rounds=4, proposer_num_explanations_per_round=2, proposer_num_x_samples_per_round=6)

    # a slightly more complex test to check if the cluster model works
    X = news_title_for_clustering
    clustering_result: ClusteringResult = explain_cluster(X, num_clusters=4, proposer_num_rounds=6, proposer_num_explanations_per_round=4, proposer_num_x_samples_per_round=6, proposer_detailed=False)
    
    print("Clustering Result on simple dataset:")
    print(simple_clustering_result)
    
    print("Clustering Result on news title dataset:")
    print(clustering_result)
