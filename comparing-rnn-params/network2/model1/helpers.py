from pytorch_metric_learning import testers
from pytorch_metric_learning.utils import accuracy_calculator

class CustomAccuracyCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_10(self, knn_labels, query_labels, **kwargs):
        if knn_labels is None:
            return 0
        return accuracy_calculator.precision_at_k(
            knn_labels, 
            query_labels[:, None], 
            10,
            self.avg_of_avgs,
            self.label_comparison_fn)

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_10"] 

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)