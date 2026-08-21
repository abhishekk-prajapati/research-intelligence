import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from backend.database import Paper

class EvaluationEngine:
    @staticmethod
    def get_ground_truth_domain(primary_category: str, title: str = "", abstract: str = "") -> str:
        """
        Determines the ground truth domain based strictly on arXiv categories
        and standard keyword checks for reinforcement learning.
        """
        cat = primary_category.lower() if primary_category else ""
        text = f"{title} {abstract}".lower()

        # 1. Computer Vision
        if any(c in cat for c in ["cs.cv", "cs.gr", "eess.iv"]):
            return "Computer Vision"
        
        # 2. Natural Language Processing
        if any(c in cat for c in ["cs.cl"]):
            return "Natural Language Processing"
            
        # 3. Robotics & Control
        if any(c in cat for c in ["cs.ro", "cs.sy", "eess.sy"]):
            return "Robotics & Control"
            
        # 4. Quantum Computing
        if "quant-ph" in cat:
            return "Quantum Computing"
            
        # 5. Reinforcement Learning
        if any(c in cat for c in ["cs.lg", "stat.ml", "cs.ai"]):
            if any(w in text for w in ["reinforcement learning", "policy gradient", "q-learning", "dqn", "markov decision"]):
                return "Reinforcement Learning"
                
        # 6. Machine Learning (General)
        if any(c in cat for c in ["cs.lg", "stat.ml", "cs.ai", "cs.ne"]):
            return "Machine Learning (General)"
            
        return "Machine Learning (General)"

    @classmethod
    def evaluate_classifier(cls, db_papers: list) -> dict:
        """
        Calculates classification accuracy, precision, recall, F1 score,
        per-class breakdown, confusion matrix, and mismatch analysis.
        """
        if not db_papers:
            return {
                "overall": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "per_class": {},
                "confusion_matrix": {"labels": [], "matrix": []},
                "mismatches": []
            }

        y_true = []
        y_pred = []
        mismatches = []
        details = []

        # All possible classes in sorted order for consistent matrix labeling
        all_classes = [
            "Computer Vision",
            "Natural Language Processing",
            "Reinforcement Learning",
            "Robotics & Control",
            "Quantum Computing",
            "Machine Learning (General)"
        ]

        for p in db_papers:
            true_domain = cls.get_ground_truth_domain(p.primary_category, p.title, p.abstract)
            pred_domain = p.domain if p.domain else "Machine Learning (General)"
            
            y_true.append(true_domain)
            y_pred.append(pred_domain)

            paper_detail = {
                "id": p.id,
                "title": p.title,
                "primary_category": p.primary_category,
                "true_domain": true_domain,
                "predicted_domain": pred_domain,
                "is_correct": true_domain == pred_domain
            }
            details.append(paper_detail)

            if true_domain != pred_domain:
                mismatches.append({
                    "id": p.id,
                    "title": p.title,
                    "abstract": p.abstract[:200] + "...",
                    "primary_category": p.primary_category,
                    "true_domain": true_domain,
                    "predicted_domain": pred_domain
                })

        # Calculate metrics using scikit-learn
        accuracy = float(accuracy_score(y_true, y_pred))
        
        # Macro average metrics
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0.0
        )
        
        # Weighted average metrics
        weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0.0
        )

        # Per-class metrics
        unique_labels = sorted(list(set(y_true + y_pred)))
        # Filter all_classes to only include those present in the labels to avoid zero-division issues,
        # but actually confusion_matrix can handle labels parameter
        
        prec_list, rec_list, f1_list, support_list = precision_recall_fscore_support(
            y_true, y_pred, labels=all_classes, zero_division=0.0
        )

        per_class = {}
        for idx, label in enumerate(all_classes):
            per_class[label] = {
                "precision": float(prec_list[idx]),
                "recall": float(rec_list[idx]),
                "f1": float(f1_list[idx]),
                "support": int(support_list[idx])
            }

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        
        return {
            "overall": {
                "accuracy": accuracy,
                "macro": {
                    "precision": float(macro_prec),
                    "recall": float(macro_rec),
                    "f1": float(macro_f1)
                },
                "weighted": {
                    "precision": float(weighted_prec),
                    "recall": float(weighted_rec),
                    "f1": float(weighted_f1)
                }
            },
            "per_class": per_class,
            "confusion_matrix": {
                "labels": all_classes,
                "matrix": cm.tolist()
            },
            "mismatches": mismatches[:50] # Cap mismatches to top 50
        }
