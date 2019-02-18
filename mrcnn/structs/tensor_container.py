class TensorContainer():
    def to(self, device):  # !pylint: disable=C0103
        """Apply pytorch's to() to all tensors in this container."""
        for key, value in self.__dict__.items():
            self.__dict__[key] = value.to(device)
        return self

    def cpu(self):
        """Apply pytorch's cpu() to all tensors in this container."""
        for key, value in self.__dict__.items():
            self.__dict__[key] = value.cpu()
        return self

    def numpy(self):
        """Apply pytorch's numpy() to all tensors in this container."""
        for key, value in self.__dict__.items():
            self.__dict__[key] = value.numpy()
        return self

    def __str__(self):
        to_str = ''
        for key, tensor in self.__dict__.items():
            to_str += ' ' + key + ': ' + str(tensor.shape)
        return to_str

    def __len__(self):
        for _, tensor in self.__dict__.items():
            return tensor.shape[0]
