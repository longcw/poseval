import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import argparse
from collections import OrderedDict

from pyposeeval.evaluateAP import evaluateAP
from pyposeeval.evaluateTracking import evaluateTracking
from pyposeeval import eval_helpers
from pyposeeval.eval_helpers import Joint


def evaluate(ground_truth_dir, predictions_dir, output_dir, eval_pose, eval_tracking, save_per_seq=False):
    argv = ['', ground_truth_dir, predictions_dir]

    print("Loading data")
    gtFramesAll, prFramesAll = eval_helpers.load_data_dir(argv)

    print("# gt frames  :", len(gtFramesAll))
    print("# pred frames:", len(prFramesAll))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_dict = OrderedDict()

    def update_results(headers, values):
        headers = [s.strip().replace('\\', '') for s in headers.split('&')]
        values = [s.strip().replace('\\', '') for s in values.split('&')]
        for header, value in zip(headers, values):
            if len(header) > 0:
                results_dict[header] = float(value)

    if eval_pose:
        #####################################################
        # evaluate per-frame multi-person pose estimation (AP)

        # compute AP
        print("Evaluation of per-frame multi-person pose estimation")
        apAll, preAll, recAll = evaluateAP(gtFramesAll, prFramesAll, output_dir, True, save_per_seq)

        # print AP
        print("Average Precision (AP) metric:")
        headers, values = eval_helpers.printTable(apAll)
        update_results(headers, values)

    if eval_tracking:
        #####################################################
        # evaluate multi-person pose tracking in video (MOTA)

        # compute MOTA
        print("Evaluation of video-based  multi-person pose tracking")
        metricsAll = evaluateTracking(gtFramesAll, prFramesAll, output_dir, True, save_per_seq)

        metrics = np.zeros([Joint().count + 4, 1])
        for i in range(Joint().count + 1):
            metrics[i, 0] = metricsAll['mota'][0, i]
        metrics[Joint().count + 1, 0] = metricsAll['motp'][0, Joint().count]
        metrics[Joint().count + 2, 0] = metricsAll['pre'][0, Joint().count]
        metrics[Joint().count + 3, 0] = metricsAll['rec'][0, Joint().count]

        # print AP
        print("Multiple Object Tracking (MOT) metrics:")
        headers, values = eval_helpers.printTable(metrics, motHeader=True)

    return results_dict, results_dict.get('Total', 0)


if __name__ == "__main__":
    def parseArgs():
        parser = argparse.ArgumentParser(description="Evaluation of Pose Estimation and Tracking (PoseTrack)")
        parser.add_argument("-g", "--groundTruth", required=False, type=str,
                            help="Directory containing ground truth annotatations per sequence in json format")
        parser.add_argument("-p", "--predictions", required=False, type=str,
                            help="Directory containing predictions per sequence in json format")
        parser.add_argument("-e", "--evalPoseEstimation", required=False, action="store_true",
                            help="Evaluation of per-frame  multi-person pose estimation using AP metric")
        parser.add_argument("-t", "--evalPoseTracking", required=False, action="store_true",
                            help="Evaluation of video-based  multi-person pose tracking using MOT metrics")
        parser.add_argument("-s", "--saveEvalPerSequence", required=False, action="store_true",
                            help="Save evaluation results per sequence", default=False)
        parser.add_argument("-o", "--outputDir", required=False, type=str, help="Output directory to save the results",
                            default="./out")
        return parser.parse_args()

    def main():
        args = parseArgs()
        print(args)
        evaluate(args.groundTruth, args.predictions, args.outputDir,
                 args.evalPoseEstimation, args.evalPoseTracking, args.saveEvalPerSequence)

    main()