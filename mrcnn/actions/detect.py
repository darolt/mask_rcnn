
import datetime
import os

from mrcnn.utils import visualize
from mrcnn.utils.rle import mask_to_rle


def detect(model, dataset, results_dir):
    """Run detection on images in the given directory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Running on {}".format(dataset.dataset_dir))

    # Create directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(results_dir, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image])[0]
        r = {k: v.detach().cpu().numpy() for k, v in r.items()}
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        img = visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        img.savefig("{}/{}.png".format(submit_dir,
                                       dataset.image_info[image_id]["id"]))
        plt.close()

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)
