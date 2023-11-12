# basically copied and amended the Ignite tutorial to set up the basic experiment

import torch

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import global_step_from_engine


def set_up_and_run_training(model,run_prefix,train_loader,val_loader,max_epoch,earlystoping_patience,log_interval=50,device=None,):

    if not device in ('cuda','cpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logs={"train_iter":list(),
          "train_epoch":list(),
          "valid_epoch":list(),}
    
    model=model.to(device)

    optimizer = model.configure_optimizers()
    criterion = model.pretty_loss

    main_trainer = create_supervised_trainer(model, optimizer, criterion, device)

    val_metrics = {"loss": Loss(criterion)}

    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")
        logs["train_iter"].append((engine.state.epoch,engine.state.iteration,engine.state.output))

    main_trainer.add_event_handler(Events.ITERATION_COMPLETED(every=log_interval),log_training_loss)

    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")
        logs["train_epoch"].append((trainer.state.epoch,metrics['loss']))

    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")
        logs["valid_epoch"].append((trainer.state.epoch,metrics['loss']))

    main_trainer.add_event_handler(Events.EPOCH_COMPLETED,log_training_results)
    main_trainer.add_event_handler(Events.EPOCH_COMPLETED,log_validation_results)


    def score_function(engine):
        return engine.state.metrics["loss"]


    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=1,
        filename_prefix=run_prefix,
        score_function=score_function,
        score_name="loss",
        require_empty=False,
        global_step_transform=global_step_from_engine(main_trainer),
    )

    es_handler = EarlyStopping(patience=earlystoping_patience, score_function=lambda x: -1*score_function(x), trainer=main_trainer)
    
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    val_evaluator.add_event_handler(Events.COMPLETED, es_handler)
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    main_trainer.run(train_loader, max_epochs=max_epoch)

    return logs

def dummy_event_test():
    help(Events.COMPLETED)
    
        
        