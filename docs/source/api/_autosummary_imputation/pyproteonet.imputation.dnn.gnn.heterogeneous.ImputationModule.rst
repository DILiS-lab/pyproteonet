pyproteonet.imputation.dnn.gnn.heterogeneous.ImputationModule
=============================================================

.. currentmodule:: pyproteonet.imputation.dnn.gnn.heterogeneous

.. autoclass:: ImputationModule
   :members:

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~ImputationModule.__init__
      ~ImputationModule.add_module
      ~ImputationModule.all_gather
      ~ImputationModule.apply
      ~ImputationModule.backward
      ~ImputationModule.bfloat16
      ~ImputationModule.buffers
      ~ImputationModule.children
      ~ImputationModule.clip_gradients
      ~ImputationModule.compile
      ~ImputationModule.compute_loss
      ~ImputationModule.configure_callbacks
      ~ImputationModule.configure_gradient_clipping
      ~ImputationModule.configure_model
      ~ImputationModule.configure_optimizers
      ~ImputationModule.configure_sharded_model
      ~ImputationModule.cpu
      ~ImputationModule.cuda
      ~ImputationModule.double
      ~ImputationModule.eval
      ~ImputationModule.extra_repr
      ~ImputationModule.float
      ~ImputationModule.forward
      ~ImputationModule.freeze
      ~ImputationModule.get_buffer
      ~ImputationModule.get_extra_state
      ~ImputationModule.get_parameter
      ~ImputationModule.get_submodule
      ~ImputationModule.half
      ~ImputationModule.ipu
      ~ImputationModule.load_from_checkpoint
      ~ImputationModule.load_state_dict
      ~ImputationModule.log
      ~ImputationModule.log_dict
      ~ImputationModule.lr_scheduler_step
      ~ImputationModule.lr_schedulers
      ~ImputationModule.manual_backward
      ~ImputationModule.modules
      ~ImputationModule.named_buffers
      ~ImputationModule.named_children
      ~ImputationModule.named_modules
      ~ImputationModule.named_parameters
      ~ImputationModule.on_after_backward
      ~ImputationModule.on_after_batch_transfer
      ~ImputationModule.on_before_backward
      ~ImputationModule.on_before_batch_transfer
      ~ImputationModule.on_before_optimizer_step
      ~ImputationModule.on_before_zero_grad
      ~ImputationModule.on_fit_end
      ~ImputationModule.on_fit_start
      ~ImputationModule.on_load_checkpoint
      ~ImputationModule.on_predict_batch_end
      ~ImputationModule.on_predict_batch_start
      ~ImputationModule.on_predict_end
      ~ImputationModule.on_predict_epoch_end
      ~ImputationModule.on_predict_epoch_start
      ~ImputationModule.on_predict_model_eval
      ~ImputationModule.on_predict_start
      ~ImputationModule.on_save_checkpoint
      ~ImputationModule.on_test_batch_end
      ~ImputationModule.on_test_batch_start
      ~ImputationModule.on_test_end
      ~ImputationModule.on_test_epoch_end
      ~ImputationModule.on_test_epoch_start
      ~ImputationModule.on_test_model_eval
      ~ImputationModule.on_test_model_train
      ~ImputationModule.on_test_start
      ~ImputationModule.on_train_batch_end
      ~ImputationModule.on_train_batch_start
      ~ImputationModule.on_train_end
      ~ImputationModule.on_train_epoch_end
      ~ImputationModule.on_train_epoch_start
      ~ImputationModule.on_train_start
      ~ImputationModule.on_validation_batch_end
      ~ImputationModule.on_validation_batch_start
      ~ImputationModule.on_validation_end
      ~ImputationModule.on_validation_epoch_end
      ~ImputationModule.on_validation_epoch_start
      ~ImputationModule.on_validation_model_eval
      ~ImputationModule.on_validation_model_train
      ~ImputationModule.on_validation_model_zero_grad
      ~ImputationModule.on_validation_start
      ~ImputationModule.optimizer_step
      ~ImputationModule.optimizer_zero_grad
      ~ImputationModule.optimizers
      ~ImputationModule.parameters
      ~ImputationModule.predict_dataloader
      ~ImputationModule.predict_step
      ~ImputationModule.prepare_data
      ~ImputationModule.print
      ~ImputationModule.register_backward_hook
      ~ImputationModule.register_buffer
      ~ImputationModule.register_forward_hook
      ~ImputationModule.register_forward_pre_hook
      ~ImputationModule.register_full_backward_hook
      ~ImputationModule.register_full_backward_pre_hook
      ~ImputationModule.register_load_state_dict_post_hook
      ~ImputationModule.register_module
      ~ImputationModule.register_parameter
      ~ImputationModule.register_state_dict_pre_hook
      ~ImputationModule.requires_grad_
      ~ImputationModule.save_hyperparameters
      ~ImputationModule.set_extra_state
      ~ImputationModule.setup
      ~ImputationModule.share_memory
      ~ImputationModule.state_dict
      ~ImputationModule.teardown
      ~ImputationModule.test_dataloader
      ~ImputationModule.test_step
      ~ImputationModule.to
      ~ImputationModule.to_empty
      ~ImputationModule.to_onnx
      ~ImputationModule.to_torchscript
      ~ImputationModule.toggle_optimizer
      ~ImputationModule.train
      ~ImputationModule.train_dataloader
      ~ImputationModule.training_step
      ~ImputationModule.transfer_batch_to_device
      ~ImputationModule.type
      ~ImputationModule.unfreeze
      ~ImputationModule.untoggle_optimizer
      ~ImputationModule.val_dataloader
      ~ImputationModule.validation_step
      ~ImputationModule.xpu
      ~ImputationModule.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~ImputationModule.CHECKPOINT_HYPER_PARAMS_KEY
      ~ImputationModule.CHECKPOINT_HYPER_PARAMS_NAME
      ~ImputationModule.CHECKPOINT_HYPER_PARAMS_TYPE
      ~ImputationModule.T_destination
      ~ImputationModule.automatic_optimization
      ~ImputationModule.call_super_init
      ~ImputationModule.current_epoch
      ~ImputationModule.device
      ~ImputationModule.dtype
      ~ImputationModule.dump_patches
      ~ImputationModule.example_input_array
      ~ImputationModule.fabric
      ~ImputationModule.global_rank
      ~ImputationModule.global_step
      ~ImputationModule.hparams
      ~ImputationModule.hparams_initial
      ~ImputationModule.local_rank
      ~ImputationModule.logger
      ~ImputationModule.loggers
      ~ImputationModule.on_gpu
      ~ImputationModule.trainer
      ~ImputationModule.training
   
   