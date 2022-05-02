from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from torch.optim.lr_scheduler import MultiStepLR
from frames_dataset import DatasetRepeater


def random_scale(kp_params, scale):
    theta = torch.rand(kp_params['fg_kp'].shape[0], 2) * (2 * scale) + (1 - scale)
    theta = torch.diag_embed(theta).unsqueeze(1).type(kp_params['fg_kp'].type())
    new_kp_params = {'fg_kp': torch.matmul(theta, kp_params['fg_kp'].unsqueeze(-1)).squeeze(-1)}
    return new_kp_params


def train_avd(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, 
              avd_network, checkpoint, log_dir, dataset):
    train_params = config['train_avd_params']

    optimizer = torch.optim.Adam(avd_network.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, kp_detector=kp_detector,
                                      bg_predictor=bg_predictor, avd_network=avd_network,
                                      dense_motion_network= dense_motion_network,optimizer_avd=optimizer)
        start_epoch = 0
    else:
        raise AttributeError("Checkpoint should be specified for mode='train_avd'.")

    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], 
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            avd_network.train()
            for x in dataloader:
                with torch.no_grad():
                    kp_source = kp_detector(x['source'].cuda())
                    kp_driving_gt = kp_detector(x['driving'].cuda())
                    kp_driving_random = random_scale(kp_driving_gt, scale=train_params['random_scale'])
                rec = avd_network(kp_source, kp_driving_random)

                reconstruction_kp = train_params['lambda_shift'] * \
                                       torch.abs(kp_driving_gt['fg_kp'] - rec['fg_kp']).mean()
                
                loss_dict = {'rec_kp': reconstruction_kp}
                loss = reconstruction_kp
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in loss_dict.items()}
                logger.log_iter(losses=losses)

            # Visualization
            avd_network.eval()
            with torch.no_grad():
                source = x['source'][:6].cuda()
                driving = torch.cat([x['driving'][[0, 1]].cuda(), source[[2, 3, 2, 1]]], dim=0)
                kp_source = kp_detector(source)
                kp_driving = kp_detector(driving)

                out = avd_network(kp_source, kp_driving)
                kp_driving = out
                dense_motion = dense_motion_network(source_image=source, kp_driving=kp_driving,
                                            kp_source=kp_source)
                generated = inpainting_network(source, dense_motion)

                generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

            scheduler.step(epoch)
            model_save = {
                'inpainting_network': inpainting_network,
                'dense_motion_network': dense_motion_network,
                'kp_detector': kp_detector,
                'avd_network': avd_network,
                'optimizer_avd': optimizer
            }
            if bg_predictor :
                model_save['bg_predictor'] = bg_predictor

            logger.log_epoch(epoch, model_save,
                             inp={'source': source, 'driving': driving},
                             out=generated)
