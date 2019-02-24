

class ProgressBar():
    INIT_VALUE = 1

    def __init__(self, total):
        """
        Args:
            total: total number of steps.
        """
        self.total = total
        self.step = ProgressBar.INIT_VALUE

    def print(self, losses):
        """
        Call in a loop to create terminal progress bar.
        Args:
            losses: Loss object
        """
        losses = losses.item()
        length = 10
        fill = '█'
        decimals = 1
        suffix = ("loss: {:.4f}, rpn_class: {:.4f}, "
                  + "rpn_bbox: {:.4f}, mrcnn_class: {:.4f}, "
                  + "mrcnn_bbox: {:.4f}, mrcnn_mask: {:.4f}")
        suffix = suffix.format(losses.total, losses.rpn_class,
                               losses.rpn_bbox, losses.mrcnn_class,
                               losses.mrcnn_bbox, losses.mrcnn_mask)

        percent = ("{0:." + str(decimals) + "f}")
        percent = percent.format(100 * (self.step / float(self.total)))

        filled_length = int(length * self.step // self.total)
        progression_bar = fill * filled_length + '-' * (length - filled_length)
        prefix = "{}/{}".format(self.step, self.total)
        print('\r%s |%s| %s%% %s' % (prefix, progression_bar, percent, suffix),
              end='\n')
        # Print New Line on Complete
        if self.step == self.total:
            self.step = ProgressBar.INIT_VALUE
            print()

        self.step += 1
