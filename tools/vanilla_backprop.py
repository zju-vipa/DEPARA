class VanillaBackprop():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()

    def feature_extract(self, x, part, layer):

        for name, module in self.model.features._modules.items():

            x = module(x)
            if part == 'feature' and name == layer:

                return x

        x = x.view(x.size(0), -1)
        for name, module in self.model.classifier._modules.items():
            x = module(x)

            if part == 'classifier' and name == layer:

                return x

        return x

    def generate_gradients(self, train_data, layer, part, device):
        # Zero grads
        model_output = self.feature_extract(train_data, part, layer)
        self.model.zero_grad()
        model_output.mean().backward()
        gradients_as_arr = train_data.grad.data.cpu().numpy()
        return gradients_as_arr


if __name__ == '__main__':
    pass
