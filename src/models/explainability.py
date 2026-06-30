class AutomataExplainer:
    """Olasılıksal Otomata kararlarını akademik ve izlenebilir loğlara dönüştürür."""
    
    @staticmethod
    def generate_log(t, current_state, incoming_pattern, status, mapped_to, distance, prob, cumulative_path_prob, decision, confidence_score, transition_history, total_exits, counterfactuals=None, similarity_report=None):
        """Her adım için detaylı bir açıklanabilirlik loğu üretir."""
        
        reasoning = (
            f"Gecis olasiligi ({prob:.4f}) esik degerden buyuk oldugu icin normal." 
            if decision == "normal" 
            else f"Gecis olasiligi ({prob:.4f}) esik degerin altinda kaldigi icin anomali!"
        )
        if status == "unseen":
            reasoning += f" Ayrica veri unseen oldugu icin Levenshtein ile {mapped_to} durumuna (Mesafe: {distance}) eslendi."

        if isinstance(transition_history, (dict, set)):
            safe_history = list(transition_history)
        elif transition_history is None:
            safe_history = []
        else:
            safe_history = list(transition_history)

        formatted_history = []
        for item in safe_history[-5:]:
            if isinstance(item, tuple):
                formatted_history.append(f"{item[0]} -> {item[1]}")
            else:
                formatted_history.append(str(item))

        return {
            "time_step": t,
            "state": current_state,                  
            "current_state_frequency": int(total_exits),
            "pattern": incoming_pattern,             
            "status": status,
            "mapped_to": mapped_to,
            "distance": distance,
            "transition_probability": float(prob),
            "cumulative_path_probability": float(cumulative_path_prob),
            "decision": decision,
            "confidence_score": float(confidence_score),
            "decision_reason": reasoning,
            "transition_history": formatted_history,
            "counterfactual_analysis": counterfactuals if counterfactuals is not None else [],
            "similarity_analysis": similarity_report if similarity_report is not None else []
        }