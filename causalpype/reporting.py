class Reporter:
    
    def __init__(self):
        self.artifacts = []

    def add_figure(self, name: str, fig):
        self.artifacts.append(('figure', name, fig))
    
    def generate_html(self):
        pass

    