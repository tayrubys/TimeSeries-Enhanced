class PSTNode:
    def __init__(self):
        self.children = {}
        self.counts = {}

    def total_count(self):
        return sum(self.counts.values())


class ProbabilisticSuffixTree:
    def __init__(self, max_depth=3, min_count=2, smoothing=True, smoothing_alpha=1.0):
        self.max_depth = max_depth #gecmise ne kadar bakicagimizi belirler
        self.min_count = min_count #belirli bir contextin minimum kac kez gectigini belirler
        self.smoothing = smoothing
        self.smoothing_alpha = smoothing_alpha
        self.root = PSTNode()
        self.vocabulary = set() # modelin gordugu tum sembollerin seti

    def fit(self, sequence):
        if len(sequence) < 2:
            raise ValueError("PST egitimi icin en az 2 pattern gereklidir.")

        self.vocabulary = set(sequence)#benzersiz sembollerin setini olusturur
        self.root = PSTNode()#her eğitimde agacin rootunu sifirlar
        #verilen sequence uzerinde agaci olusturur (ilki geciyorux cunku gecmisi yok)
        for i in range(1, len(sequence)):
            next_symbol = sequence[i]

            self.root.counts[next_symbol] = self.root.counts.get(next_symbol, 0) + 1

            max_context = min(self.max_depth, i)#siniri asmamak için max derşnligi belirler
            #farklı derinliklerdeki geçmişleri (suffix'leri) ağaca ekliyor
            for depth in range(1, max_context + 1):
                context = sequence[i - depth:i]
                self._add_context(context, next_symbol)
    
    #istediğimiz geçişleri seçip ayrı ayrı normal PST’ye veya anomaly PST’ye ekleyebilmemizi sağlar
    def add_observation(self, history, next_symbol):
        # Bu geçişte görülen sembolleri vocabulary içine ekler ve vocabulary smoothing hesaplamasında kullanılır
        self.vocabulary.add(next_symbol)

        for symbol in history:
            self.vocabulary.add(symbol)

        #root node geçmişe bakmadan genel geçiş dağılımını tutar
        self.root.counts[next_symbol] = self.root.counts.get(next_symbol, 0) + 1

        max_context = min(self.max_depth, len(history))

        #farklı uzunluktaki suffix/context bilgilerini ağaca ekle
        for depth in range(1, max_context + 1):
            context = history[-depth:]
            self._add_context(context, next_symbol)   

    def _add_context(self, context, next_symbol):
        node = self.root#Verilen context'i ağaçta oluşturup sayacını artır
        #agaca geriye dogru suffix kurma
        for symbol in reversed(context):
            if symbol not in node.children:
                node.children[symbol] = PSTNode()
            node = node.children[symbol]

        node.counts[next_symbol] = node.counts.get(next_symbol, 0) + 1
    #aradığı context'i bulmak için agacı dolaşır ve node'u döndürür
    def _find_node(self, context):
        node = self.root

        for symbol in reversed(context):
            if symbol not in node.children:
                return None # bu gecmıs yoksa
            node = node.children[symbol]

        return node
    #en uzun geçmişi(context) bulur ve node'u döndürür
    def find_best_context(self, history):
        max_context = min(self.max_depth, len(history))
        #en uzun geçmişten başlayarak en kısa geçmişe kadar bakar ve uygun node'u bulur
        for depth in range(max_context, 0, -1):
            context = history[-depth:]
            node = self._find_node(context)

            if node is not None and node.total_count() >= self.min_count:
                return context, node
        #hicbir sey yoksa root'u döndür
        return [], self.root

    def predict_probability(self, history, next_symbol):
        context, node = self.find_best_context(history)

        total = node.total_count()
        vocab_size = max(1, len(self.vocabulary))
        #Smoothing aktifse Laplace formülünü uygula
        if self.smoothing:
            count = node.counts.get(next_symbol, 0)
            return (count + self.smoothing_alpha) / (
                total + self.smoothing_alpha * vocab_size
            )
        #smoothing aktif değilse, sadece olasılığı hesapla
        if total == 0:
            return 0.0

        return node.counts.get(next_symbol, 0) / total
    
    #pst ağacındaki toplam context/node ve geçiş sayılarını hesaplama
    def count_nodes_and_transitions(self):
        def traverse(node):
            # Bulunduğumuz node'u da sayıyoruz.
            node_count = 1

            #node altında kaç farklı hedef sembol gözlenmişse,o kadar gecis var kabul et
            transition_count = len(node.counts)

            #child nodeları da gezerek tüm ağacı say
            for child in node.children.values():
                child_nodes, child_transitions = traverse(child)
                node_count += child_nodes
                transition_count += child_transitions

            return node_count, transition_count

        return traverse(self.root)    