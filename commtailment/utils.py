import math
import scipy
import scipy.io
import pandas as pd
import simbench as sb
import pandapower as pp
import numpy as np
import networkx as nx
import tqdm

import commtailment


CHARGING_EFFICIENCY = 0.9


def to_MW(Wh):
    # assumes 15 min intervals
    watt = Wh * 4
    MW = watt / 1e6
    return MW


def to_Wh(MW):
    # assumes 15 min intervals
    watt = MW * 1e6
    Wh = watt / 4
    return Wh


def p_to_q(_p):
    return math.sqrt((_p / 0.95) ** 2 - _p ** 2)


def convert_profiles_to_simbench_dict(profile_mapping, net):
    '''
    OUTPUT: python dict with keys:
    dict_keys([
        ('load', 'p_mw'),
        ('load', 'q_mvar'),
        ('sgen', 'p_mw'),
        ('gen', 'p_mw'),
        ('storage', 'p_mw')])
    and a pandas DataFrame as value for each tuple.
    The DataFrame columns must correspond to the index of the respective
    element DataFrame (net.element). The DataFrame index represents the
    timesteps'''
    dict_keys = [
        ('load', 'p_mw'),
        ('load', 'q_mvar'),
        ('sgen', 'p_mw'),
        ('gen', 'p_mw'),
        ('storage', 'p_mw'),
    ]
    index = profile_mapping[net.load.loc[0].bus]['meter_records'].index
    # convert RangeIndex to Int64Index so pandapower understands it
    index = list(index)
    simbench_dict = {key: pd.DataFrame(index=index) for key in dict_keys[:-1]}
    for idx, load in net.load.iterrows():
        bus = load.bus
        load_p_mw = profile_mapping[bus]['meter_records']['p_mw_load']
        simbench_dict[('load', 'p_mw')][idx] = load_p_mw
        load_p_mvar = profile_mapping[bus]['meter_records']['q_mvar_load']
        simbench_dict[('load', 'q_mvar')][idx] = load_p_mvar
    for idx, sgen in net.sgen.iterrows():
        bus = sgen.bus
        p_mw_sgen = profile_mapping[bus]['meter_records']['p_mw_sgen']
        simbench_dict[('sgen', 'p_mw')][idx] = p_mw_sgen
    return simbench_dict


def apply_absolute_values(net, absolute_values_dict, case_or_time_step):
    ''' apply the state of all electrical components at
     "case_or_time_step" to the net. '''
    for elm_param in absolute_values_dict.keys():
        profiles_df = absolute_values_dict[elm_param]
        if profiles_df.shape[1]:
            elm = elm_param[0]
            param = elm_param[1]
            net[elm].loc[:, param] = profiles_df.loc[case_or_time_step]


def distribute_pv_sgens(net, pv_ratio):
    # randomly disribute static generators in net.
    n_loads = len(net.load)
    n_sgens = round(pv_ratio * n_loads)
    sgens = pd.DataFrame(columns=net.sgen.columns)
    load_bus_ids = list(net.load.bus)
    sgen_bus_ids = np.random.choice(load_bus_ids, n_sgens, replace=False)
    for i, sgen_bus_id in enumerate(sgen_bus_ids):
        values = net.sgen.iloc[0].copy()
        # change name
        name = values['name']
        name = ' '.join(name.split()[:-1]) + ' {}'.format(i)
        values['name'] = name
        # change bus
        values['bus'] = sgen_bus_id
        sgens.loc[i] = values
    net.sgen = sgens.astype(net.sgen.dtypes)
    return sgen_bus_ids


def make_linear_communities(net, community_size):
    ''' make list of communites. Community = list of bus_ids within that
    community. Traverse each branch bus by bus and add them to a
    community until it's full or until a graph leaf is reached
    (so no communities are created, that are not connected directly). '''
    if community_size < 1:
        raise RuntimeError
    elif community_size == 1:
        return [[bus] for bus in list(net.load.bus)]
    bus = net.trafo.lv_bus[0]
    graph = pp.topology.create_nxgraph(net)
    branches_first_buses = (
        set(nx.all_neighbors(graph, bus)) - set([net.trafo.hv_bus[0]])
    )
    communities = []
    done_buses = {bus}
    for branch_first_bus in branches_first_buses:
        community = [branch_first_bus]
        done_buses.add(branch_first_bus)
        open_ends = list(
            set(nx.all_neighbors(graph, branch_first_bus)) - done_buses
        )
        while len(open_ends) > 0:
            bus = None
            # ensure bus is connected to a load
            while bus not in list(net.load.bus):
                bus = open_ends.pop()
                done_buses.add(bus)
                new_ends = list(set(nx.all_neighbors(graph, bus)) - done_buses)
                open_ends += new_ends
            community.append(bus)
            if len(community) == community_size or not new_ends:
                communities.append(community)
                community = []
        if community and community not in communities:
            communities.append(community)
    return communities


def map_profiles(net, _profiles):
    # map a profile to each load/sgen bus. Select them randomly
    all_load_bus_ids = list(net.load.bus)
    selected_profiles = np.random.choice(
        _profiles, len(net.load.bus), replace=False
    )
    return dict(zip(all_load_bus_ids, selected_profiles))


def get_community_installed_power(community, profile_mapping, sgen_bus_ids):
    # add up installed power of all sgens in that community
    installed_power = 0
    for bus in community:
        if bus in sgen_bus_ids:
            installed_power += profile_mapping[bus]['installed_power_MW']
    return installed_power


def curtail(feed_in_limit, communities, profile_mapping, sgen_bus_ids):
    # curtail all pv profiles for sgens in the net community-wise
    if feed_in_limit == 1:
        return 0
    losses_Wh = 0
    for community in communities:
        community_sgen_bus_ids = list(set(community) & set(sgen_bus_ids))
        community_load_bus_ids = list(set(community) - set(sgen_bus_ids))
        # assert community == community_sgen_bus_ids + community_load_bus_ids
        installed_power = sum(
            [profile_mapping[bus]['installed_power_MW']
             for bus in community_sgen_bus_ids]
        )
        allowed_power = installed_power * feed_in_limit
        # use NEGATIVE values for loads in balance
        community_load_profiles = [
            - profile_mapping[bus]['meter_records']['p_mw_load']
            for bus in community_load_bus_ids
        ]
        community_sgen_profiles = [
            profile_mapping[
                bus
            ]['meter_records']['demand_and_feed_balance_p_mw']
            for bus in community_sgen_bus_ids
        ]
        # balance is a pandas Series of p_mw values for each timestep
        alle = community_load_profiles + community_sgen_profiles
        balance = sum(alle)
        balance = balance[balance > 0]
        # cut off peaks, that exceed the allowed power. aka curtailment
        # of the community as a whole
        curtailed_balance = balance.clip(upper=allowed_power)
        losses_mw = balance - curtailed_balance
        losses_Wh += to_Wh(sum(losses_mw))

        # total production of community
        total_production = sum(community_sgen_profiles)
        # get the factors for each timestep of how much all
        # sgens need to be curtailed to match curtailment regulations
        curtailment_factors = losses_mw / total_production
        # factor of how much energy is allowed to be fed in by all
        remaining_factors = curtailment_factors.apply(lambda x: 1-x)
        # due to division by 0 curtailment_factors may have nan values
        # since there is nothing to be curtailed there relpacing nan
        # with 0 is fine
        for bus in community_sgen_bus_ids:
            # curtail each sgen producing too much energy by multiplying
            # with its communities excess factor
            df = profile_mapping[bus]['meter_records']
            df['p_mw_sgen'] = df['p_mw_sgen'].multiply(
                remaining_factors, fill_value=1
            )
    return losses_Wh


def make_feed_demand_balance(
    sgen_bus_ids,
    profile_mapping,
    battery_size_kWh_per_kWp,
    prediction_based,
    feed_in_limit,
):
    ''' calculate the resulting balance after charging/discharging
    batteries including energy feed in and demand. Return charging
    losses'''
    total_charging_losses_Wh = 0
    for bus in sgen_bus_ids:
        df = profile_mapping[bus]['meter_records']
        after_self_consumption = (
            df['energy_production [Wh]'] - df['energy_consumption [Wh]']
        )
        if battery_size_kWh_per_kWp is not None:
            # simulate demand and feed in with batteries installed
            after_battery_MW, charging_losses_Wh = charge_battery(
                bus,
                profile_mapping,
                after_self_consumption,
                battery_size_kWh_per_kWp,
                prediction_based,
                feed_in_limit,
            )
            df['demand_and_feed_balance_p_mw'] = after_battery_MW
            total_charging_losses_Wh += charging_losses_Wh
        else:
            df['demand_and_feed_balance_p_mw'] = after_self_consumption
    return total_charging_losses_Wh


def charge_battery(
    bus,
    profile_mapping,
    after_self_consumption,
    battery_size_kWh_per_kWp,
    prediction_based,
    feed_in_limit,
):
    '''calculate new feed-in/demand profile that takes batteries
    into account. Return charging losses.'''
    charging_losses_Wh = 0
    installed_power_MW = profile_mapping[bus]['installed_power_MW']
    installed_power_kWp = installed_power_MW * 1e3
    battery_size_kWh = battery_size_kWh_per_kWp * installed_power_kWp
    battery_size_Wh = battery_size_kWh * 1e3
    if prediction_based:
        availability_profile_Wh = make_availability_profile(
            after_self_consumption,
            feed_in_limit,
            installed_power_MW,
            battery_size_Wh,
        )
    else:
        availability_profile_Wh = [
            battery_size_Wh
        ] * len(after_self_consumption)
    battery_state_Wh = 0
    after_battery_MW = []
    for i, (_, energy_Wh) in enumerate(after_self_consumption.iteritems()):
        if energy_Wh > 0:
            # charge battery and loose 10 %
            would_charge = energy_Wh * CHARGING_EFFICIENCY
            available_capacity = max(
                0,
                availability_profile_Wh[i] - battery_state_Wh,
            )
            if would_charge > available_capacity:
                # full
                feed_in_Wh = energy_Wh * (
                    1 - available_capacity / would_charge
                )
                loss = (energy_Wh - feed_in_Wh) * (1 - CHARGING_EFFICIENCY)
                if battery_state_Wh < available_capacity:
                    battery_state_Wh = available_capacity
            else:
                feed_in_Wh = 0
                loss = energy_Wh * (1 - CHARGING_EFFICIENCY)
                battery_state_Wh += would_charge
            after_battery_MW.append(to_MW(feed_in_Wh))
            charging_losses_Wh += loss
        elif energy_Wh < 0:
            # energy_Wh and would_discharge ARE NEGATIVE!!!!!
            would_discharge = energy_Wh / CHARGING_EFFICIENCY
            if abs(would_discharge) > battery_state_Wh:
                # empty
                demand_Wh = energy_Wh * (
                    1 - battery_state_Wh / abs(would_discharge)
                )
                loss = abs(
                    energy_Wh - demand_Wh
                ) * (1 - CHARGING_EFFICIENCY)
                battery_state_Wh = 0
            else:
                demand_Wh = 0
                loss = abs(energy_Wh * (1 - CHARGING_EFFICIENCY))
                battery_state_Wh += would_discharge
            after_battery_MW.append(to_MW(demand_Wh))
            charging_losses_Wh += loss
        else:
            after_battery_MW.append(0)
    return after_battery_MW, charging_losses_Wh


def make_availability_profile(
    after_self_consumption_Wh,
    feed_in_limit,
    installed_power_MW,
    battery_size_Wh,
):
    ''' calculate the specific available capacity for each time step.
    Assume perfect prediction for load and production.
    '''
    feed_in_limit_MW = feed_in_limit * installed_power_MW
    feed_in_limit_Wh = to_Wh(feed_in_limit_MW)

    def _prepare(Wh):
        # filter feed_in_limit and apply charging inefficiencies.
        if Wh > 0:
            Wh = max(0, Wh - feed_in_limit_Wh)
            # capacity needed is less than energy excess
            Wh *= CHARGING_EFFICIENCY
        else:
            # capacity being reduced is more than demand
            Wh /= CHARGING_EFFICIENCY
        return Wh
    balance_profile_Wh = after_self_consumption_Wh.apply(_prepare)
    # reversed cumsum with minimum and maximum
    availability_profile_Wh = [battery_size_Wh]
    for Wh in balance_profile_Wh[::-1]:
        available = availability_profile_Wh[-1] - Wh
        available = max(0, min(battery_size_Wh, available))
        availability_profile_Wh.append(available)
    return availability_profile_Wh[::-1][1:]


def get_installed_power(profile):
    ''' my best estimate so far is simply the highest power excluding
    2 per mille outliers. '''
    return profile['meter_records']['p_mw_sgen'].quantile(0.998)


def prepare_profiles(use_one_for_all_sgen=None):
    # load profiles, convert to power, find out installed power of PV modules
    print("Reading from matlab")
    service, private = load_reasonable_profiles()
    all_profiles = service + private
    print("converting to MW and getting peak power")
    for profile in all_profiles:
        records_df = profile['meter_records']
        records_df['p_mw_sgen'] = records_df[
            'energy_production [Wh]'
        ].apply(to_MW)
        records_df['p_mw_load'] = records_df[
            'energy_consumption [Wh]'
        ].apply(to_MW)
        records_df['q_mvar_load'] = records_df['p_mw_load'].apply(p_to_q)
        profile['installed_power_MW'] = get_installed_power(profile)
    ''' use only one profile scaled by the peak power value of the others
    to simulate a uniform location'''
    print("scaling")
    if use_one_for_all_sgen is not None:
        ref_profile = all_profiles[use_one_for_all_sgen]
        ref_max = ref_profile['installed_power_MW']
        for profile in all_profiles:
            factor = profile['installed_power_MW'] / ref_max
            records_df = profile['meter_records']
            records_df['p_mw_sgen'] = factor * ref_profile[
                'meter_records'
            ]['p_mw_sgen']
            records_df['energy_production [Wh]'] = factor * ref_profile[
                'meter_records'
            ]['energy_production [Wh]']
    return service, private


def prepare_experiment(
    mixed_profiles,
    simbench_code,
    pv_ratio,
    feed_in_limit,
    community_size,
    seed,
    battery_size_kWh_per_kWp=None,
    prediction_based=False,
):
    # seed for reproducible randomness
    np.random.seed(seed)
    # load simbench network
    net = sb.get_simbench_net(simbench_code)
    # distribute pv generators in net according to pv_ratio.
    sgen_bus_ids = distribute_pv_sgens(net, pv_ratio)
    # make communities with max community_size members
    # should be list of lists of bus ids
    # (with bus being the grid connection point of load/sgen)
    communities = make_linear_communities(net, community_size)
    # distribute profiles on network. Just map them by IDs. For loads and sgens
    profile_mapping = map_profiles(net, mixed_profiles)
    # make profile of resulting feed-in and demand energy after charging
    charging_losses_Wh = make_feed_demand_balance(
        sgen_bus_ids,
        profile_mapping,
        battery_size_kWh_per_kWp,
        prediction_based,
        feed_in_limit,
    )
    # curtail the profiles community wise (sgens)
    losses_Wh = curtail(
        feed_in_limit, communities, profile_mapping, sgen_bus_ids
    )
    absolute_values_dict = convert_profiles_to_simbench_dict(
        profile_mapping, net
    )
    total_production = sum(
        [
            sum(
                profile_mapping[bus]['meter_records']['energy_production [Wh]']
                )
            for bus in net.sgen.bus
        ]
    )
    return (
        net,
        absolute_values_dict,
        profile_mapping,
        charging_losses_Wh,
        losses_Wh,
        total_production,
    )


def community_stat(communities, profile_mapping, sgen_bus_ids):
    # get the average installed power, number of PV houses and
    # demand per community
    total_power = 0
    total_consumption = 0
    total_production = 0
    total_number_members = 0
    for community in communities:
        community_sgen_bus_ids = list(set(community) & set(sgen_bus_ids))
        installed_power_MW = sum(
            [profile_mapping[bus]['installed_power_MW']
             for bus in community_sgen_bus_ids]
        )
        community_load_profiles = [
            profile_mapping[bus]['meter_records']['energy_consumption [Wh]']
            for bus in community
        ]
        community_sgen_profiles = [
            profile_mapping[bus]['meter_records']['energy_production [Wh]']
            for bus in community_sgen_bus_ids
        ]
        total_power += installed_power_MW
        total_consumption += sum(community_load_profiles)
        total_production += sum(community_sgen_profiles)
        total_number_members += len(community)
    average_power = total_power / len(communities)
    average_consumption = sum(total_consumption) / len(communities)
    average_production = sum(total_production) / len(communities)
    average_number_members = total_number_members / len(communities)
    return [
        average_power,
        average_consumption,
        average_production,
        average_number_members,
    ]


def validate_net(net, absolute_values_dict, time_steps_range):
    time_steps = range(*time_steps_range)
    for time_step in tqdm.tqdm(time_steps):
        # print('Processing:', time_step, end='\r')
        apply_absolute_values(net, absolute_values_dict, time_step)
        try:
            pp.runpp(net, numba=True)
        except pp.powerflow.LoadflowNotConverged:
            print("\nFAIL! (not converging): ", time_step)
            return False, 'not converging', time_step
        else:
            valid, reason = validate_results(net)
            if not valid:
                print("\nFAIL! (", reason, '): ', time_step)
                return False, reason, time_step
    print('SUCCESS!')
    return True, None, -1


def validate_results(net):
    # check line loading
    if (net.res_line.loading_percent > 100).any():
        print("Line Overload:")
        print(net.res_line[net.res_line.loading_percent > 100])
        return False, "line overload"
    # check bus voltages between 0.9 and 1.1 pu
    if (net.res_bus.vm_pu > 1.1).any():
        print("Voltage too high:")
        print(net.res_bus[net.res_bus.vm_pu > 1.1])
        return False, 'bus voltage too high'
    if (net.res_bus.vm_pu < 0.9).any():
        print("Voltage too low:")
        print(net.res_bus[net.res_bus.vm_pu < 0.9])
        return False, 'bus voltage too low'
    # check trafo loading
    if (net.res_trafo.loading_percent > 100).any():
        print("Trafo Overload:")
        print(net.res_trafo[net.res_trafo.loading_percent > 100])
        return False, "trafo overload"
    return True, None


def load_reasonable_profiles():
    service, private, header = _load_mat_lab_files()
    service = _from_mat_to_reasonable(service, header)
    private = _from_mat_to_reasonable(private, header)
    return service, private


def _load_mat_lab_files():

    # load mat lab files
    service_mat = scipy.io.loadmat(commtailment.config.PROFILE_SERVICE)
    private_mat = scipy.io.loadmat(commtailment.config.PROFILE_PRIVATE)
    header_mat = scipy.io.loadmat(commtailment.config.PROFILE_HEADER)

    # get rid of the header
    service = service_mat['profile_service']
    private = private_mat['profile_private']
    header = header_mat['profile_header']
    return service, private, header


def _from_mat_to_reasonable(_profiles, header):
    print('number households:', len(_profiles))
    reasonable = []
    cols = [weird_array[0] for weird_array in header[0][0][0]]
    for profile in _profiles:
        new_profile = {}
        # actual records np array(35139, 15)
        meter_records = pd.DataFrame(profile[0], columns=cols)
        new_profile['meter_records'] = meter_records
        new_profile['_'.join(header[0][1][0].split())] = profile[1][0]
        # completeness
        new_profile['_'.join(header[0][2][0].split())] = profile[2][0][0]
        reasonable.append(new_profile)
    return reasonable
