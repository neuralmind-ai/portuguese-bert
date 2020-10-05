import jsonlines
from argparse import Namespace
from datetime import datetime
from typing import Any

from eval_tools import SequenceMetrics


def to_float(value):
    if isinstance(value, list):
        return [float(val) for val in value]
    else:
        return float(value)


def compile_results(args: Namespace,
                    train_metrics: SequenceMetrics,
                    valid_metrics: SequenceMetrics,
                    best_epoch_metric: str = 'f1_score',
                    **extra_values: Any):
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    attrs_args = [
        ('num_train_epochs', 'epochs'),
        'learning_rate',
        'train_batch_size',
        'gradient_accumulation_steps',
        'train_file',
        'valid_file',
        'pooler',
        'freeze_bert',
        'output_dir',
        'labels_file',
        'classifier_lr',
        'no_crf',
        'seed',
        'labels_file',
        'lstm_hidden_size',
        'lstm_layers',
    ]

    for attr in attrs_args:
        if len(attr) == 2:
            source, dest = attr
        else:
            source = dest = attr
        results[dest] = getattr(args, source, None)

    best_epoch = valid_metrics.get_best_epoch(best_epoch_metric)
    results['best_epoch'] = best_epoch

    attrs_metrics = [
        'f1_score',
        'precision',
        'recall',
    ]

    for prefix, metrics in [('train', train_metrics),
                            ('valid', valid_metrics)]:
        for attr in attrs_metrics:
            key = f'{prefix}_{attr}'
            values = metrics.history.get(attr)
            if values:
                results[key] = to_float(values)
                results[f'best_{key}'] = to_float(max(values))

    results['classification_report'] = valid_metrics.get_value(
        'classification_report', best_epoch)

    for name, value in extra_values.items():
        results[name] = to_float(value)

    return results


def write_jsonl_results(results, path):
    """Append a line to a jsonlines file."""
    assert path.endswith('.jsonl')
    with jsonlines.open(path, 'a') as writer:
        writer.write(results)
