import numpy as np

class PSTNode:
    def __init__(self, symbol=None):
        self.symbol = symbol  #düğümdeki SAX sembolü
        self.children = {}    #alt dallara hızlı erişmek için dict
        self.counts = {}      #bu düğümden sonra hangi sembolden kaç tane gelmiş sayacı
    #next_symbol un gelme olasılığını hesaplama
    def get_probability(self, next_symbol, alphabet_size, smoothing=True):

        total_counts = sum(self.counts.values())
        #laplace uygulama sıfırı cozmek için
        if smoothing:
            total = total_counts + alphabet_size
            count = self.counts.get(next_symbol, 0) + 1
        else:
            if total_counts == 0:
                return 1.0 / alphabet_size
            total = total_counts
            count = self.counts.get(next_symbol, 0)
            
        return count / total

    def get_probability_distribution(self, alphabet_size):
        """
        KL-Divergence budaması için bu düğümün (bağlamın) tüm olası 
        sembollere karşı olasılık dağılımını (P veya Q) çıkarıyoruz.
        """
        dist = {}
        # SAX sembolleri genelde 'a', 'b', 'c' diye gider. 
        # ASCII tablosundan alfabeyi dinamik oluştur
        alphabet = [chr(i) for i in range(97, 97 + alphabet_size)] 
        
        for symbol in alphabet:
            dist[symbol] = self.get_probability(symbol, alphabet_size, smoothing=True)
        return dist


class ProbabilisticSuffixTree:
    def __init__(self, max_depth, alphabet_size):
        self.max_depth = max_depth
        self.alphabet_size = alphabet_size
        self.root = PSTNode() # Kök düğüm
        
    def fit(self, sax_sequence):
        """
        Eğitim verisindeki SAX dizisini ağaca beslediğimiz ana döngü.
        """
        n = len(sax_sequence)
        
        for i in range(n):
            for length in range(1, self.max_depth + 1):
                if i - length >= 0:
                    context = sax_sequence[i-length : i]
                    next_symbol = sax_sequence[i]
                    self._add_sequence(context, next_symbol)
    #suffix mantığıyla ağaçta aşağı inip sayacı güncelle
    def _add_sequence(self, context, next_symbol):
 
        current_node = self.root
        
        # En yakın geçmiş en çok etkiye sahip olduğu için context'i tersten okuyoruz.
        for symbol in reversed(context):
            if symbol not in current_node.children:
                current_node.children[symbol] = PSTNode(symbol)
            current_node = current_node.children[symbol]
            
        current_node.counts[next_symbol] = current_node.counts.get(next_symbol, 0) + 1
    #agacı egıttıten hemen sonra gereksiz dalları (KL) Iraksamasi ile budala
    def prune(self, kl_threshold=0.01):

        self._prune_node(self.root, kl_threshold)
        
    def _prune_node(self, node, kl_threshold):
        if not node.children:
            return #budanacak dal yok 
            
        #kendi olasılık dağılımımız:(Q - Ebeveyn / Daha kısa geçmiş)
        q_dist = node.get_probability_distribution(self.alphabet_size)
        children_to_remove = []
        
        for symbol, child_node in node.children.items():
            #child olasılık dağılımı:P - Alt Dal / Daha derin geçmiş
            p_dist = child_node.get_probability_distribution(self.alphabet_size)
            
            # KL Iraksamasını hesapla: D_KL(P || Q) = sum( P(x) * log(P(x) / Q(x)) )
            kl_div = 0.0
            for x in p_dist:
                p_x = p_dist[x]
                q_x = q_dist[x]
                if p_x > 0 and q_x > 0:
                    kl_div += p_x * np.log(p_x / q_x)
                    
            # Eğer alt dalın sunduğu yeni olasılık dağılımı parentten farklı değilse ve threshold'dan küçükse dalı budala
            if kl_div < kl_threshold:
                children_to_remove.append(symbol)
            else:
                #dalı kesmiyorsak, onun da alt dallarını kontrole devam et
                self._prune_node(child_node, kl_threshold)
                
        # Gereksiz çocukları (ve dolayısıyla tüm alt ağacını) acımadan sil
        for symbol in children_to_remove:
            del node.children[symbol]
    #test asamasında ağaçta inebildiğimiz en derin (budanmamış) geçmişe kadar inip olasılığı bulma
    def predict_probability(self, context, next_symbol):

        current_node = self.root
        best_prob = current_node.get_probability(next_symbol, self.alphabet_size)
        
        for symbol in reversed(context):
            if symbol in current_node.children:
                current_node = current_node.children[symbol]
                best_prob = current_node.get_probability(next_symbol, self.alphabet_size)
            else:
                break
                
        return best_prob