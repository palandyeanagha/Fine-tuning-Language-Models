import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    if args.use_wandb:
        wandb.init(
            project="nlp-hw4-t5",
            name=args.experiment_name,
            config={
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'max_n_epochs': args.max_n_epochs,
                'patience_epochs': args.patience_epochs,
                'scheduler_type': args.scheduler_type,
                'weight_decay': args.weight_decay,
                'finetune': args.finetune,
            }
        )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        # Fine-tune pretrained T5 model
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
    else:
        # Train from scratch with T5 config
        config = T5Config.from_pretrained('t5-small')
        model = T5ForConditionalGeneration(config)
    
    model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best, optimizer=None, scheduler=None, epoch=None, best_f1=None, epochs_since_improvement=None):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if best_f1 is not None:
        checkpoint['best_f1'] = best_f1
    if epochs_since_improvement is not None:
        checkpoint['epochs_since_improvement'] = epochs_since_improvement
    
    torch.save(checkpoint, checkpoint_path)

def load_model_from_checkpoint(args, best, optimizer=None, scheduler=None):
    # Load model from a checkpoint
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Initialize model first
    model = initialize_model(args)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    # Load optimizer and scheduler if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_f1 = checkpoint.get('best_f1', -1)
    epochs_since_improvement = checkpoint.get('epochs_since_improvement', 0)
    
    return model, epoch, best_f1, epochs_since_improvement

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

