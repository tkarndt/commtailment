import commtailment
import copy
import multiprocessing
import json

seed = 42
use_same_profile = 3
s, p = commtailment.utils.prepare_profiles(use_same_profile)

definitions = []
for grid in [2, 3]:
    simbench_code = '1-LV-rural{}--0-no_sw'.format(grid)
    for pv_ratio in [0.15, 0.3, 0.45]:
        for curtailment_pu in [0.7, 0.5, 0.3]:
            for community_size in range(6, 11):
                definition = [
                    simbench_code,
                    pv_ratio,
                    curtailment_pu,
                    community_size,
                    seed,
                ]
                definitions.append(definition)

for grid in [1]:
    simbench_code = '1-LV-rural{}--0-no_sw'.format(grid)
    for pv_ratio in [0.15, 0.3, 0.45]:
        for curtailment_pu in [0.7, 0.5, 0.3]:
            for community_size in range(1, 6):
                definition = [
                    simbench_code,
                    pv_ratio,
                    curtailment_pu,
                    community_size,
                    seed,
                ]
                definitions.append(definition)


#  validate
def mp_worker(definition):
    name = str(use_same_profile) + '__'
    name += str(definition)[1:-1].replace(', ', '__').replace("'", '')
    definition = [copy.deepcopy(p + s)] + definition
    print('running:', name)
    [
        net,
        absolute_values_dict,
        _,
        charging_losses_Wh,
        losses_Wh,
        total_production_Wh,
    ] = commtailment.utils.prepare_experiment(*definition)
    time_rage = (0, len(absolute_values_dict['sgen', 'p_mw'].index))

    validation = commtailment.utils.validate_net(
        net, absolute_values_dict, time_rage
    )

    with open("validation/results.json".format(name), "a+") as f:
        json.dump(
            [
                name,
                validation,
                charging_losses_Wh,
                losses_Wh,
                total_production_Wh,
            ], f)
        f.write('\n')


def mp_handler():
    # use multiprocessing to speed thinfs up
    p = multiprocessing.Pool(3)
    p.map(mp_worker, definitions)


if __name__ == '__main__':
    mp_handler()
