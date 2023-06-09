import math
import os
import random
import time

import cv2
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from loguru import logger

from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.units import Units

from StarCraft2Env import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import nest_asyncio

nest_asyncio.apply()


class Zerg_bot(BotAI):
    def __init__(self, transaction, lock):

        self.lock = lock
        self.transaction = transaction
        self.worker_supply = 12  # 农民数量
        self.army_supply = 0  # 部队人口
        self.base_count = 1  # 基地数量
        self.base_pending = 0
        self.enemy_units_count = 0
        # self.army_units_list=[]  # 我方部队列表
        # self.enemy_list = []
        self.overlord_count = 1
        self.overlord_planning_count = 0
        self.queen_count = 0
        # building_count
        self.Evolution_chamber_count = 0
        self.Spire_count = 0
        self.Roach_warren_count = 0
        self.Baneling_nest_count = 0
        self.Nydus_Network_count = 0
        self.Nydus_Worm_count = 0
        self.Greater_spire_count = 0
        self.Hydralisk_den_count = 0
        self.Lurker_Den_count = 0
        self.Lair_count = 0
        self.Hive_count = 0
        self.Ultralisk_Cavern_count = 0
        self.gas_buildings_count = 0
        self.gas_buildings_planning_count = 0
        self.overseer_count = 0
        self.rally_defend = False

    def get_information(self):
        self.worker_supply = self.workers.amount
        self.army_supply = self.supply_army
        self.base_count = self.structures(UnitTypeId.HATCHERY).amount + self.structures(
            UnitTypeId.LAIR).amount + self.structures(UnitTypeId.HIVE).amount
        self.base_pending = self.already_pending(UnitTypeId.HATCHERY)
        self.overlord_count = self.units(UnitTypeId.OVERLORD).amount
        self.overlord_planning_count = self.already_pending(UnitTypeId.OVERLORD)

        self.queen_count = self.units(UnitTypeId.QUEEN).amount
        self.Evolution_chamber_count = self.structures(UnitTypeId.EVOLUTIONCHAMBER).amount
        self.Spire_count = self.structures(UnitTypeId.SPIRE).amount
        self.Roach_warren_count = self.structures(UnitTypeId.ROACHWARREN).amount
        self.Baneling_nest_count = self.structures(UnitTypeId.BANELINGNEST).amount
        self.Nydus_Network_count = self.structures(UnitTypeId.NYDUSNETWORK).amount
        self.Nydus_Worm_count = self.structures(UnitTypeId.NYDUSWORMLAVADEATH).amount
        self.Greater_spire_count = self.structures(UnitTypeId.GREATERSPIRE).amount
        self.Hydralisk_den_count = self.structures(UnitTypeId.HYDRALISKDEN).amount
        self.Lurker_Den_count = self.structures(UnitTypeId.LURKERDEN).amount
        self.Lair_count = self.structures(UnitTypeId.LAIR).amount
        self.Hive_count = self.structures(UnitTypeId.HIVE).amount
        self.Ultralisk_Cavern_count = self.structures(UnitTypeId.ULTRALISKCAVERN).amount
        self.gas_buildings_count = self.gas_buildings.amount
        self.gas_buildings_planning_count = self.already_pending(UnitTypeId.EXTRACTOR)
        self.overseer_count = self.units(UnitTypeId.OVERSEER).amount
        return {'work_supply': self.worker_supply,  # 农民数量
                'mineral': self.minerals,  # 晶体矿
                'gas': self.vespene,  # 高能瓦斯
                'supply_left': self.supply_left,  # 剩余人口
                'army_supply': self.army_supply,  # 部队人口
                # 'army_list': self.army_units_list,  # 部队列表
                # 'enemy_list': self.enemy_list,  # 敌方单位列表
                'enemy_count': self.enemy_units_count,  # 敌方单位数量
                'game_time': self.time,
                'base_count': self.base_count,  # 我方基地数量
                'Evolution_chamber_count': self.Evolution_chamber_count,  # bv数量
                'Spire_count': self.Spire_count,  # vs数量
                'Roach_warren_count': self.Roach_warren_count,
                'Baneling_nest_count': self.Baneling_nest_count,
                'Nydus_Network_count': self.Nydus_Network_count,
                'Nydus_Worm_count': self.Nydus_Worm_count,
                'Greater_spire_count': self.Greater_spire_count,
                'Hydralisk_den_count': self.Hydralisk_den_count,
                'Lurker_Den_count': self.Lurker_Den_count,
                'Lair_count': self.Lair_count,
                'Hive_count': self.Hive_count,
                'Ultralisk_Cavern_count': self.Ultralisk_Cavern_count,
                'gas_buildings_count': self.gas_buildings_count,
                'spore crawler_count ': self.structures(UnitTypeId.SPORECRAWLER).amount,
                'spine crawler_count ': self.structures(UnitTypeId.SPINECRAWLER).amount,

                'planning_base_count': self.base_pending,  # 我方正在建造的基地数量
                'planning_gas_buildings_count': self.gas_buildings_planning_count,
                'planning_spawningpool_count': self.already_pending(UnitTypeId.SPAWNINGPOOL),
                'planning_banelingnest_count': self.already_pending(UnitTypeId.BANELINGNEST),
                'planning_roachwarren_count': self.already_pending(UnitTypeId.ROACHWARREN),
                'planning_lair_count': self.already_pending(UnitTypeId.LAIR),
                'planning_hive_count': self.already_pending(UnitTypeId.HIVE),
                'planning_infestationpit_count': self.already_pending(UnitTypeId.INFESTATIONPIT),
                'planning_hydraliskden_count': self.already_pending(UnitTypeId.HYDRALISKDEN),
                'planning_spire_count': self.already_pending(UnitTypeId.SPIRE),
                'planning_ultraliskcavern_count': self.already_pending(UnitTypeId.ULTRALISKCAVERN),
                'planning_lurkerden_count': self.already_pending(UnitTypeId.LURKERDEN),
                'planning_greatspire_count': self.already_pending(UnitTypeId.GREATERSPIRE),
                'planning_evolutuionchamber_count': self.already_pending(UnitTypeId.EVOLUTIONCHAMBER),

                'queen_count': self.queen_count,  # 我方女王数量
                'overlord_count': self.overlord_count,  # 房子数量
                'overseer_count': self.overseer_count,
                'larvea_count': self.larva.amount,
                'drone_count': self.units(UnitTypeId.DRONE).amount,
                'zergling_count': self.units(UnitTypeId.ZERGLING).amount,
                'baneing_count': self.units(UnitTypeId.BANELING).amount,
                'roach_count': self.units(UnitTypeId.ROACH).amount,
                'ravager_count': self.units(UnitTypeId.RAVAGER).amount,
                'hydralisk_count': self.units(UnitTypeId.HYDRALISK).amount,
                'lurker_count': self.units(UnitTypeId.LURKERMP).amount,
                'mutalisk_count': self.units(UnitTypeId.MUTALISK).amount,
                'corruptor_count': self.units(UnitTypeId.CORRUPTOR).amount,
                'broodlord_count': self.units(UnitTypeId.BROODLORD).amount,
                'ultralisk_count': self.units(UnitTypeId.ULTRALISK).amount,
                'infestor_count': self.units(UnitTypeId.INFESTOR).amount,
                'viper_count': self.units(UnitTypeId.VIPER).amount,
                'swarmhost_count': self.units(UnitTypeId.SWARMHOSTMP).amount,

                'planning_queen_count': self.already_pending(UnitTypeId.QUEENMP),
                'planning_drone_count': self.already_pending(UnitTypeId.DRONE),
                'overlord_planning_count': self.overlord_planning_count,  # 正在建造的房子数量
                'planning_overseer_count': self.already_pending(UnitTypeId.OVERSEER),
                'planning_zergling_count': self.already_pending(UnitTypeId.ZERGLING),
                'planning_baneing_count': self.already_pending(UnitTypeId.BANELING),
                'planning_roach_count': self.already_pending(UnitTypeId.ROACH),
                'planning_ravager_count': self.already_pending(UnitTypeId.RAVAGER),
                'planning_hydralisk_count': self.already_pending(UnitTypeId.HYDRALISK),
                'planning_lurker_count': self.already_pending(UnitTypeId.LURKERMP),
                'planning_mutalisk_count': self.already_pending(UnitTypeId.MUTALISK),
                'planning_corruptor_count': self.already_pending(UnitTypeId.CORRUPTOR),
                'planning_broodlord_count': self.already_pending(UnitTypeId.BROODLORD),
                'planning_ultralisk_count': self.already_pending(UnitTypeId.ULTRALISK),
                'planning_infestor_count': self.already_pending(UnitTypeId.INFESTOR),
                'planning_viper_count': self.already_pending(UnitTypeId.VIPER),
                'planning_swarmhost_count': self.already_pending(UnitTypeId.SWARMHOSTMP),



                }

    # def get_army_list(self):
    #     temp_list = []
    #     for unit in self.units.of_type(
    #             {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER,
    #              UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
    #              UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
    #              UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
    #              UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
    #              UnitTypeId.CHANGELINGZEALOT}):
    #         temp_list.append(unit)
    #     temp_list = list(temp_list)
    #     self.army_units_list = temp_list
    # print(self.army_units_list)
    #
    # def get_enemy_list(self):
    #     enemy_list = []
    #     enemy = self.enemy_units
    #     for _ in enemy:
    #         enemy_list.append(_)
    #     enemy_list = list(enemy_list)
    #     self.enemy_list = enemy_list
    #     print(self.enemy_list)
    async def defend(self):
        print("Defend:", self.rally_defend)
        if self.structures(UnitTypeId.HATCHERY).exists or self.structures(UnitTypeId.LAIR).exists or self.structures(
                UnitTypeId.HIVE).exists and self.supply_army >= 2:
            for base in self.townhalls:
                if self.enemy_units.amount >= 2 and self.enemy_units.closest_distance_to(base) < 30:
                    self.rally_defend = True
                    for unit in self.units.of_type(
                            {UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.QUEEN, UnitTypeId.ROACH,
                             UnitTypeId.OVERSEER,
                             UnitTypeId.ULTRALISK, UnitTypeId.MUTALISK, UnitTypeId.INFESTOR, UnitTypeId.CORRUPTOR,
                             UnitTypeId.BROODLORD,
                             UnitTypeId.OVERSEER, UnitTypeId.RAVAGER, UnitTypeId.VIPER, UnitTypeId.SWARMHOSTMP,
                             UnitTypeId.LURKER}):
                        closed_enemy = self.enemy_units.sorted(lambda x: x.distance_to(unit))
                        unit.attack(closed_enemy[0])
                else:
                    self.rally_defend = False

            if self.rally_defend == True:
                map_center = self.game_info.map_center
                rally_point = self.townhalls.random.position.towards(map_center, distance=5)
                for unit in self.units.of_type(
                        {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER, UnitTypeId.SENTRY,
                         UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
                         UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
                         UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
                         UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
                         UnitTypeId.CHANGELINGZEALOT}):
                    if unit.distance_to(self.start_location) > 100 and unit not in self.unit_tags_received_action:
                        unit.move(rally_point)

    async def on_step(self, iteration: int):
        if self.time_formatted == '00:00':

            if self.start_location == Point2((160.5, 46.5)):
                self.Location = -1  # detect location
            else:
                self.Location = 1
        information = self.get_information()
        # self.get_army_list()
        await self.defend()
        # print('iter:%d'%iteration)
        # 画图
        map = np.zeros((224, 224, 3), dtype=np.uint8)

        # 矿产
        for mineral in self.mineral_field:
            pos = mineral.position
            c = [175, 255, 255]
            fraction = mineral.mineral_contents / 1800
            if mineral.is_visible:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]
                # print(int(fraction * i) for i in c)
                # print([int(fraction * i) for i in c])
                # print(map[math.ceil(pos.y)][math.ceil(pos.x)])
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [50, 50, 50]

        # 瓦斯
        for vespene in self.vespene_geyser:
            pos = vespene.position
            c = [255, 175, 255]
            fraction = vespene.vespene_contents / 2250
            if vespene.is_visible:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [50, 50, 50]

        # 基础设施
        for structure in self.structures:
            pos = structure.position
            if structure.type_id == UnitTypeId.COMMANDCENTER:
                c = [255, 255, 175]
            else:
                c = [0, 255, 175]
            fraction = structure.health_percentage
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

        # 装备
        for unit in self.units:
            pos = unit.position
            if unit.type_id == UnitTypeId.GHOST:
                c = [255, 0, 0]
            else:
                c = [175, 255, 0]
            fraction = unit.health_percentage
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

        # 敌军出生位置
        for enemy_location in self.enemy_start_locations:
            pos = enemy_location.position
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [0, 0, 255]
        # 敌军设施
        for structure in self.enemy_structures:
            pos = structure.position
            c = [0, 100, 255]
            fraction = structure.health_percentage
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]
        # 敌军d单位
        for unit in self.enemy_units:
            pos = unit.position
            c = [100, 0, 255]
            fraction = unit.health_percentage
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]

        cv2.imshow('map', cv2.flip(cv2.resize(map, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), 0))
        cv2.waitKey(1)
        self.lock.acquire()
        self.transaction['observation'] = map
        self.transaction['information'] = information

        self.lock.release()

        while self.transaction['action'] is None:
            time.sleep(0.001)
        action = self.transaction['action']

        await self.distribute_workers()
        reward = 0
        if action == 0:
            larvae: Units = self.larva

            print(f'action={action}')

            if self.townhalls.exists and self.units(UnitTypeId.DRONE).exists and larvae:
                if '00:00' <= self.time_formatted <= '06:00':
                    if self.supply_left <= 3 and self.already_pending(
                            UnitTypeId.OVERLORD) <= 2 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.OVERLORD):
                            larvae.random.train(UnitTypeId.OVERLORD)
                            print('train overlord')
                if '06:00' <= self.time_formatted <= '07:00':
                    if self.supply_left <= 5 and self.already_pending(
                            UnitTypeId.OVERLORD) <= 4 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.OVERLORD):
                            larvae.random.train(UnitTypeId.OVERLORD)
                            print('train overlord')

                if '06:00' <= self.time_formatted <= '08:00':
                    if self.supply_left <= 5 and self.already_pending(
                            UnitTypeId.OVERLORD) <= 3 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.OVERLORD):
                            larvae.random.train(UnitTypeId.OVERLORD)
                            print('train overlord')

                    if self.supply_left <= 7 and self.already_pending(
                            UnitTypeId.OVERLORD) <= 4 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.OVERLORD):
                            larvae.random.train(UnitTypeId.OVERLORD)
                            print('train overlord')
                if '10:00' <= self.time_formatted:
                    if self.supply_left <= 7 and self.already_pending(
                            UnitTypeId.OVERLORD) <= 4 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.OVERLORD):
                            larvae.random.train(UnitTypeId.OVERLORD)
                            print('train overlord')
        elif action == 1:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.townhalls.exists and larvae:
                if self.workers.amount + self.already_pending(UnitTypeId.DRONE) <= 80 and self.supply_left >= 0:
                    if self.can_afford(UnitTypeId.DRONE) and larvae:
                        larvae.random.train(UnitTypeId.DRONE)
                        reward +=0.00001/(int(self.time)+1)
                        print('train drones')
        elif action == 2:
            print(f'action={action}')
            if self.units(UnitTypeId.OVERLORD).exists and self.units(
                    UnitTypeId.DRONE).exists and self.townhalls.exists:
                # gas
                for base in self.townhalls:
                    for vespene in self.vespene_geyser.closer_than(10, base):
                        if self.can_afford(UnitTypeId.EXTRACTOR) and not self.structures(
                                UnitTypeId.EXTRACTOR).closer_than(2, vespene):
                            await self.build(UnitTypeId.EXTRACTOR, vespene)
                            have_builded = True
                            print('build extractor')
        elif action == 3:
            if self.units(UnitTypeId.DRONE).exists:

                print(f'action={action}')
                if self.time_formatted <= "06:00":
                    if self.townhalls.amount + self.already_pending(UnitTypeId.HATCHERY) <= 4 and self.can_afford(
                            UnitTypeId.HATCHERY):
                        await self.expand_now()
                if "06:00" <= self.time_formatted <= "10:00":
                    if self.townhalls.amount + self.already_pending(UnitTypeId.HATCHERY) <= 6 and self.can_afford(
                            UnitTypeId.HATCHERY):
                        await self.expand_now()
                if "10:00" <= self.time_formatted:
                    if self.townhalls.amount + self.already_pending(UnitTypeId.HATCHERY) <= 10 and self.can_afford(
                            UnitTypeId.HATCHERY):
                        await self.expand_now()
                    print('build hatchery')
        elif action == 4:
            print(f'action={action}')
            if self.units(UnitTypeId.OVERLORD).exists and self.units(
                    UnitTypeId.DRONE).exists and self.townhalls.exists:
                if not self.structures(UnitTypeId.SPAWNINGPOOL).exists and not self.already_pending(
                        UnitTypeId.SPAWNINGPOOL):
                    worker_candidates = self.workers.filter(lambda worker: (
                                                                                   worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)

                    place_postion = self.start_location.position + Point2((self.Location * 10, 0))
                    placement_position = await self.find_placement(UnitTypeId.SPAWNINGPOOL, near=place_postion,
                                                                   placement_step=6)
                    if placement_position and self.can_afford(UnitTypeId.SPAWNINGPOOL) and self.already_pending(
                            UnitTypeId.SPAWNINGPOOL) == 0:
                        build_worker = worker_candidates.closest_to(placement_position)
                        build_worker.build(UnitTypeId.SPAWNINGPOOL, placement_position)
                if self.structures(UnitTypeId.SPAWNINGPOOL).exists:
                    spawningpool = self.structures(UnitTypeId.SPAWNINGPOOL).random
                    abilities = await self.get_available_abilities(spawningpool)
                    if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.already_pending(
                            UpgradeId.ZERGLINGMOVEMENTSPEED) == 0:
                        if self.can_afford(
                                UpgradeId.ZERGLINGMOVEMENTSPEED) and spawningpool.is_idle and AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST in abilities:
                            spawningpool.research(UpgradeId.ZERGLINGMOVEMENTSPEED)
                            print('research zergling speed')

                    if self.structures(UnitTypeId.SPAWNINGPOOL).ready and self.already_pending(
                            UpgradeId.ZERGLINGATTACKSPEED) == 0:
                        if self.can_afford(
                                UpgradeId.ZERGLINGATTACKSPEED) and spawningpool.is_idle and AbilityId.RESEARCH_ZERGLINGADRENALGLANDS in abilities:
                            spawningpool.research(UpgradeId.ZERGLINGATTACKSPEED)
                            print('research zergling attack speed')



        elif action == 5:
            print(f'action={action}')
            if self.units(UnitTypeId.OVERLORD).exists and self.units(
                    UnitTypeId.DRONE).exists and self.townhalls.exists:
                if self.structures(UnitTypeId.SPAWNINGPOOL).exists and not self.structures(
                        UnitTypeId.BANELINGNEST).exists \
                        and self.already_pending(UnitTypeId.BANELINGNEST) == 0:
                    worker_candidates = self.workers.filter(lambda worker: (
                                                                                   worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)

                    place_postion = self.start_location.position + Point2((self.Location * 10, 0))
                    placement_position = await self.find_placement(UnitTypeId.BANELINGNEST, near=place_postion,
                                                                   placement_step=6)
                    if placement_position and self.can_afford(UnitTypeId.BANELINGNEST) and self.already_pending(
                            UnitTypeId.BANELINGNEST) == 0:
                        build_worker = worker_candidates.closest_to(placement_position)
                        build_worker.build(UnitTypeId.BANELINGNEST, placement_position)
                        print('build banelingnest')

                if self.structures(UnitTypeId.BANELINGNEST).exists:

                    banelingnest = self.structures(UnitTypeId.BANELINGNEST).random
                    abilities = await self.get_available_abilities(banelingnest)

                    if self.structures(UnitTypeId.BANELINGNEST).ready and self.already_pending(
                            UpgradeId.CENTRIFICALHOOKS) == 0:
                        if self.can_afford(
                                UpgradeId.CENTRIFICALHOOKS) and banelingnest.is_idle and AbilityId.RESEARCH_CENTRIFUGALHOOKS in abilities:
                            banelingnest.research(UpgradeId.CENTRIFICALHOOKS)
                            print('research baneling speed')

        elif action == 6:
            print(f'action={action}')
            if self.units(UnitTypeId.OVERLORD).exists and self.units(
                    UnitTypeId.DRONE).exists and self.townhalls.exists \
                    and self.structures(UnitTypeId.SPAWNINGPOOL).exists:
                if not self.structures(UnitTypeId.ROACHWARREN).exists and not self.already_pending(
                        UnitTypeId.ROACHWARREN):
                    worker_candidates = self.workers.filter(lambda worker: (
                                                                                   worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)

                    place_postion = self.start_location.position + Point2((self.Location * 10, 0))
                    placement_position = await self.find_placement(UnitTypeId.ROACHWARREN, near=place_postion,
                                                                   placement_step=6)
                    if placement_position and self.can_afford(UnitTypeId.ROACHWARREN) and self.structures(
                            UnitTypeId.ROACHWARREN).amount + self.already_pending(UnitTypeId.ROACHWARREN) == 0:
                        build_worker = worker_candidates.closest_to(placement_position)
                        build_worker.build(UnitTypeId.ROACHWARREN, placement_position)
            if self.structures(UnitTypeId.ROACHWARREN).exists:
                roachwarren = self.structures(UnitTypeId.ROACHWARREN).random
                abilities = await self.get_available_abilities(roachwarren)
                if self.structures(UnitTypeId.LAIR).ready and self.already_pending(
                        UpgradeId.GLIALRECONSTITUTION) == 0:
                    if self.can_afford(
                            UpgradeId.GLIALRECONSTITUTION) and roachwarren.is_idle and AbilityId.RESEARCH_GLIALREGENERATION in abilities:
                        roachwarren.research(UpgradeId.GLIALRECONSTITUTION)
                        print('research roach speed')
                if self.structures(UnitTypeId.LAIR).ready and self.already_pending(
                        UpgradeId.TUNNELINGCLAWS) == 0:
                    if self.can_afford(
                            UpgradeId.TUNNELINGCLAWS) and roachwarren.is_idle and AbilityId.RESEARCH_TUNNELINGCLAWS in abilities:
                        roachwarren.research(UpgradeId.TUNNELINGCLAWS)
                        print('research tunnelingclaws')

        elif action == 7:
            print(f'action={action}')
            if self.townhalls.exists:
                base = self.townhalls.random
                bases = self.townhalls
                abilities = await self.get_available_abilities(base)
                if self.townhalls.ready and self.already_pending(
                        UpgradeId.OVERLORDSPEED) == 0:
                    if self.can_afford(
                            UpgradeId.OVERLORDSPEED) and AbilityId.RESEARCH_PNEUMATIZEDCARAPACE in abilities:
                        for hatchery in bases:
                            if hatchery.is_idle:
                                hatchery.research(UpgradeId.OVERLORDSPEED)
                                print('research overlord speed')

        elif action == 8:
            print(f'action={action}')
            if self.structures(UnitTypeId.LAIR).exists:
                base = self.townhalls.random
                bases = self.townhalls
                abilities = await self.get_available_abilities(base)
                if self.structures(UnitTypeId.LAIR).ready and self.already_pending(UpgradeId.BURROW) == 0:
                    if self.can_afford(UpgradeId.BURROW) and AbilityId.RESEARCH_BURROW in abilities:
                        for hatchery in bases:
                            if hatchery.is_idle:
                                hatchery.research(UpgradeId.BURROW)
                                print('research burrow')
        elif action == 9:
            if self.units(UnitTypeId.OVERLORD).exists and self.units(
                    UnitTypeId.DRONE).exists and self.townhalls.exists \
                    and self.structures(UnitTypeId.LAIR).exists or self.structures(UnitTypeId.HIVE).exists:
                if not self.structures(UnitTypeId.HYDRALISKDEN).amount + self.already_pending(
                        UnitTypeId.HYDRALISKDEN) == 0:
                    worker_candidates = self.workers.filter(lambda worker: (
                                                                                   worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)

                    place_postion = self.start_location.position + Point2((self.Location * 10, 0))
                    placement_position = await self.find_placement(UnitTypeId.HYDRALISKDEN, near=place_postion,
                                                                   placement_step=6)
                    if placement_position and self.can_afford(UnitTypeId.HYDRALISKDEN) and self.structures(
                            UnitTypeId.HYDRALISKDEN).amount + self.already_pending(UnitTypeId.ROACHWARREN) == 0:
                        build_worker = worker_candidates.closest_to(placement_position)
                        build_worker.build(UnitTypeId.HYDRALISKDEN, placement_position)
            if self.structures(UnitTypeId.HYDRALISKDEN).exists:
                hydraden = self.structures(UnitTypeId.HYDRALISKDEN).random
                abilities = await self.get_available_abilities(hydraden)
                if self.structures(UnitTypeId.LAIR).ready and self.already_pending(
                        UpgradeId.EVOLVEMUSCULARAUGMENTS) == 0:
                    if self.can_afford(
                            UpgradeId.EVOLVEMUSCULARAUGMENTS) and hydraden.is_idle and AbilityId.RESEARCH_MUSCULARAUGMENTS in abilities:
                        hydraden.research(UpgradeId.EVOLVEMUSCULARAUGMENTS)
                        print('research hydra speed')
                if self.structures(UnitTypeId.LAIR).ready and self.already_pending(
                        UpgradeId.EVOLVEGROOVEDSPINES) == 0:
                    if self.can_afford(
                            UpgradeId.EVOLVEGROOVEDSPINES) and hydraden.is_idle and AbilityId.RESEARCH_GROOVEDSPINES in abilities:
                        hydraden.research(UpgradeId.EVOLVEGROOVEDSPINES)
                        print('research hydra range')


        elif action == 10:
            print(f'action={action}')
            if self.structures(UnitTypeId.HATCHERY).exists and self.units(UnitTypeId.DRONE).exists and self.structures(
                    UnitTypeId.SPAWNINGPOOL).exists:
                if self.already_pending(UnitTypeId.LAIR) + self.structures(UnitTypeId.LAIR).amount == 0 \
                        and self.structures(UnitTypeId.HIVE).amount + self.already_pending(UnitTypeId.HIVE) == 0:
                    if self.can_afford(UnitTypeId.LAIR):
                        bases = self.townhalls
                        for base in bases:
                            if base.is_idle and self.can_afford(UnitTypeId.LAIR):
                                if self.already_pending(UnitTypeId.LAIR) + self.structures(UnitTypeId.LAIR).amount == 0:
                                    base.build(UnitTypeId.LAIR)
                                    print('build lair')
            if self.structures(UnitTypeId.LAIR).exists and self.units(UnitTypeId.DRONE).exists and self.structures(
                    UnitTypeId.INFESTATIONPIT).exists:
                if self.structures(UnitTypeId.LAIR).exists:
                    lairs = self.structures(UnitTypeId.lair)
                    if self.can_afford(UnitTypeId.HIVE):
                        for lair in lairs:
                            if lair.is_idle:
                                if self.structures(UnitTypeId.INFESTATIONPIT).ready and self.already_pending(
                                        UnitTypeId.HIVE) + self.structures(UnitTypeId.HIVE).amount == 0:
                                    lair.build(UnitTypeId.HIVE)
                                    print('build hive')
        elif action == 11:
            print(f'action={action}')
            if self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.units(
                    UnitTypeId.DRONE).exists and self.townhalls.amount>=3:
                if self.already_pending(UnitTypeId.EVOLUTIONCHAMBER) + self.structures(
                        UnitTypeId.EVOLUTIONCHAMBER).amount <= 2:
                    if self.can_afford(UnitTypeId.EVOLUTIONCHAMBER) and self.already_pending(
                            UnitTypeId.EVOLUTIONCHAMBER) <= 2:
                        building_place = self.townhalls.random.position
                        placement_position = await self.find_placement(UnitTypeId.EVOLUTIONCHAMBER,
                                                                       near=building_place,
                                                                       placement_step=4)
                        if placement_position is not None:
                            await self.build(UnitTypeId.EVOLUTIONCHAMBER, near=placement_position)
                            print('build evolutionchamber')
            if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready:
                bvs = self.structures(UnitTypeId)
                abilities = await self.get_available_abilities(bvs)
                for bv in bvs:
                    if bv.is_idle:
                        if self.can_afford(
                                UpgradeId.ZERGMELEEWEAPONSLEVEL1) and bv.is_idle and AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL1 in abilities and self.already_pending(
                            UpgradeId.ZERGMELEEWEAPONSLEVEL1) == 0:
                            bv.research(UpgradeId.ZERGMELEEWEAPONSLEVEL1)
                            print('zerg melee plus 1 ')
                        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready and self.already_pending(
                                UpgradeId.ZERGMELEEWEAPONSLEVEL2) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGMELEEWEAPONSLEVEL2) and bv.is_idle and AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL2 in abilities:
                                bv.research(UpgradeId.ZERGMELEEWEAPONSLEVEL2)
                                print('zerg melee plus 2 ')
                        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready and self.already_pending(
                                UpgradeId.ZERGMELEEWEAPONSLEVEL3) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGMELEEWEAPONSLEVEL3) and bv.is_idle and AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL3 in abilities:
                                bv.research(UpgradeId.ZERGMELEEWEAPONSLEVEL3)
                                print('zerg melee plus 3 ')
        elif action == 12:
            print(f'action={action}')
            if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready:
                bvs = self.structures(UnitTypeId.EVOLUTIONCHAMBER)
                abilities = await self.get_available_abilities(bvs)

                for bv in bvs:
                    if bv.is_idle:
                        if self.can_afford(
                                UpgradeId.ZERGMISSILEWEAPONSLEVEL1) and bv.is_idle and AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL1 in abilities and self.already_pending(
                            UpgradeId.ZERGMISSILEWEAPONSLEVEL1) == 0:
                            bv.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL1)

                            print('zerg missile plus 1 ')
                        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready and self.already_pending(
                                UpgradeId.ZERGMISSILEWEAPONSLEVEL2) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGMISSILEWEAPONSLEVEL2) and bv.is_idle and AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL2 in abilities:
                                bv.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL2)
                                print('zerg missile plus 2 ')
                        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready and self.already_pending(
                                UpgradeId.ZERGMISSILEWEAPONSLEVEL3) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGMISSILEWEAPONSLEVEL3) and bv.is_idle and AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL3 in abilities:
                                bv.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL3)
                                print('zerg missile plus 3 ')
        elif action == 13:
            print(f'action={action}')
            if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready:
                bvs = self.structures(UnitTypeId.EVOLUTIONCHAMBER)
                abilities = await self.get_available_abilities(bvs)

                for bv in bvs:
                    if bv.is_idle:
                        if self.can_afford(
                                UpgradeId.ZERGGROUNDARMORSLEVEL1) and bv.is_idle and AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL1 in abilities and self.already_pending(
                            UpgradeId.ZERGGROUNDARMORSLEVEL1) == 0:
                            bv.research(UpgradeId.ZERGGROUNDARMORSLEVEL1)

                            print('zerg ground armor plus 1 ')
                        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready and self.already_pending(
                                UpgradeId.ZERGGROUNDARMORSLEVEL2) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGGROUNDARMORSLEVEL2) and bv.is_idle and AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL2 in abilities:
                                bv.research(UpgradeId.ZERGGROUNDARMORSLEVEL2)
                                print('zerg ground armor plus 2 ')
                        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready and self.already_pending(
                                UpgradeId.ZERGGROUNDARMORSLEVEL3) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGGROUNDARMORSLEVEL3) and bv.is_idle and AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL3 in abilities:
                                bv.research(UpgradeId.ZERGGROUNDARMORSLEVEL3)
                                print('zerg ground armor plus 3 ')
        elif action == 14:
            print(f'action={action}')
            if self.structures(UnitTypeId.LAIR).exists or self.structures(UnitTypeId.HIVE).exists and self.units(
                    UnitTypeId.DRONE).exists and self.townhalls.exists:
                if self.already_pending(UnitTypeId.INFESTATIONPIT) + self.structures(
                        UnitTypeId.INFESTATIONPIT).amount == 0:
                    if self.can_afford(UnitTypeId.INFESTATIONPIT) and self.already_pending(
                            UnitTypeId.INFESTATIONPIT) == 0:
                        building_place = self.townhalls.random.position
                        placement_position = await self.find_placement(UnitTypeId.INFESTATIONPIT,
                                                                       near=building_place,
                                                                       placement_step=4)
                        if placement_position is not None:
                            await self.build(UnitTypeId.INFESTATIONPIT, near=placement_position)
                            print('build vi')
            if self.structures(UnitTypeId.INFESTATIONPIT).ready:
                vis = self.structures(UnitTypeId)
                abilities = await self.get_available_abilities(vis)
                for vi in vis:
                    if vi.is_idle:
                        if self.can_afford(
                                UpgradeId.INFESTORENERGYUPGRADE) and vi.is_idle and AbilityId.RESEARCH_PATHOGENGLANDS in abilities and self.already_pending(
                            UpgradeId.INFESTORENERGYUPGRADE) == 0:
                            vi.research(UpgradeId.INFESTORENERGYUPGRADE)
                            print('research infestor energy upgrade')
                        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready and vi.is_idle and self.already_pending(
                                UpgradeId.NEURALPARASITE) == 0:
                            if self.can_afford(
                                    UpgradeId.NEURALPARASITE) and vi.is_idle and AbilityId.RESEARCH_NEURALPARASITE in abilities:
                                vi.research(UpgradeId.NEURALPARASITE)
                                print('research neural parasite ')

        elif action == 15:
            print(f'action={action}')
            if self.units(UnitTypeId.OVERLORD).exists and self.units(UnitTypeId.DRONE).exists and self.structures(
                    UnitTypeId.LAIR).exists or self.structures(UnitTypeId.HIVE).exists:
                if not self.structures(UnitTypeId.SPIRE).exists and self.already_pending(
                        UnitTypeId.SPIRE) + self.structures(UnitTypeId.SPIRE).amount < 2:
                    if self.can_afford(UnitTypeId.SPIRE) and self.already_pending(
                            UnitTypeId.SPIRE) <= 1 and self.structures(UnitTypeId.HIVE).exists or self.structures(
                        UnitTypeId.LAIR).exists:
                        building_place = self.townhalls.random.position
                        placement_position = await self.find_placement(UnitTypeId.SPIRE,
                                                                       near=building_place,
                                                                       placement_step=4)
                        if placement_position is not None:
                            await self.build(UnitTypeId.SPIRE, near=placement_position)
                            print('build spire')
            if self.structures(UnitTypeId.SPIRE).exists:
                spires = self.structures(UnitTypeId.SPIRE)
                abilities = await self.get_available_abilities(spires)
                for spire in spires:
                    if spire.is_idle:
                        if self.structures(UnitTypeId.SPIRE).ready and self.can_afford(
                                UpgradeId.ZERGFLYERWEAPONSLEVEL1) and spire.is_idle and AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL1 in abilities and self.already_pending(
                            UpgradeId.ZERGFLYERWEAPONSLEVEL1) == 0:
                            spire.research(UpgradeId.ZERGFLYERWEAPONSLEVEL1)

                            print('zerg fly attack plus 1 ')
                        if self.structures(UnitTypeId.SPIRE).ready and self.already_pending(
                                UpgradeId.ZERGFLYERWEAPONSLEVEL2) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGFLYERWEAPONSLEVEL2) and spire.is_idle and AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL2 in abilities:
                                spire.research(UpgradeId.ZERGFLYERWEAPONSLEVEL2)
                                print('zerg fly attack plus 2 ')
                        if self.structures(UnitTypeId.SPIRE).ready and self.already_pending(
                                UpgradeId.ZERGFLYERWEAPONSLEVEL3) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGFLYERWEAPONSLEVEL3) and spire.is_idle and AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL3 in abilities:
                                spire.research(UpgradeId.ZERGFLYERWEAPONSLEVEL3)
                                print('zerg fly attack plus 3 ')
                        if self.structures(UnitTypeId.SPIRE).ready and self.can_afford(
                                UpgradeId.ZERGFLYERARMORSLEVEL1) and spire.is_idle and AbilityId.RESEARCH_ZERGFLYERARMORLEVEL1 in abilities and self.already_pending(
                            UpgradeId.ZERGFLYERARMORSLEVEL1) == 0:
                            spire.research(UpgradeId.ZERGFLYERARMORSLEVEL1)

                            print('zerg fly armor plus 1 ')
                        if self.structures(UnitTypeId.SPIRE).ready and self.already_pending(
                                UpgradeId.ZERGFLYERARMORSLEVEL2) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGFLYERARMORSLEVEL2) and spire.is_idle and AbilityId.RESEARCH_ZERGFLYERARMORLEVEL2 in abilities:
                                spire.research(UpgradeId.ZERGFLYERARMORSLEVEL2)
                                print('zerg fly armor plus 2 ')
                        if self.structures(UnitTypeId.SPIRE).ready and self.already_pending(
                                UpgradeId.ZERGFLYERARMORSLEVEL1) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGFLYERARMORSLEVEL3) and spire.is_idle and AbilityId.RESEARCH_ZERGFLYERARMORLEVEL3 in abilities:
                                spire.research(UpgradeId.ZERGFLYERARMORSLEVEL3)
                                print('zerg fly armor plus 3 ')
        elif action == 16:
            print(f'action={action}')
            if self.structures(UnitTypeId.SPIRE).exists and self.units(UnitTypeId.DRONE).exists and self.structures(
                    UnitTypeId.HIVE).exists and self.structures(UnitTypeId.GREATERSPIRE).amount + self.already_pending(
                UnitTypeId.GREATERSPIRE) == 0:
                if self.structures(UnitTypeId.GREATERSPIRE).amount + self.already_pending(
                        UnitTypeId.GREATERSPIRE) == 0:
                    spires = self.structures(UnitTypeId.SPIRE)
                    abilities = await self.get_available_abilities(spires)
                    if self.structures(UnitTypeId.SPIRE).ready:
                        for spire in spires:
                            if spire.is_idle and self.structures(UnitTypeId.GREATERSPIRE).amount + self.already_pending(
                                    UnitTypeId.GREATERSPIRE) == 0:
                                if self.can_afford(
                                        UnitTypeId.GREATERSPIRE) and AbilityId.UPGRADETOGREATERSPIRE_GREATERSPIRE in abilities:
                                    spire.build(UnitTypeId.GREATERSPIRE)
                                    print('build greater spire')
            if self.structures(UnitTypeId.GREATERSPIRE).exists:
                greaterspires = self.structures(UnitTypeId.GREATERSPIRE)
                abilities = await self.get_available_abilities(greaterspires)
                for greaterspire in greaterspires:
                    if greaterspire.is_idle:
                        if self.structures(UnitTypeId.GREATERSPIRE).ready and self.can_afford(
                                UpgradeId.ZERGFLYERWEAPONSLEVEL1) and greaterspire.is_idle and AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL1 in abilities and self.already_pending(
                            UpgradeId.ZERGFLYERWEAPONSLEVEL1) == 0:
                            greaterspire.research(UpgradeId.ZERGFLYERWEAPONSLEVEL1)

                            print('research zerg fly attack plus 1 ')
                        if self.structures(UnitTypeId.GREATERSPIRE).ready and self.already_pending(
                                UpgradeId.ZERGFLYERWEAPONSLEVEL2) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGFLYERWEAPONSLEVEL2) and greaterspire.is_idle and AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL2 in abilities:
                                greaterspire.research(UpgradeId.ZERGFLYERWEAPONSLEVEL2)
                                print('research zerg fly attack plus 2 ')
                        if self.structures(UnitTypeId.SPIRE).ready and self.already_pending(
                                UpgradeId.ZERGFLYERWEAPONSLEVEL3) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGFLYERWEAPONSLEVEL3) and greaterspire.is_idle and AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL3 in abilities:
                                greaterspire.research(UpgradeId.ZERGFLYERWEAPONSLEVEL3)
                                print('research zerg fly attack plus 3 ')
                        if self.structures(UnitTypeId.GREATERSPIRE).ready and self.can_afford(
                                UpgradeId.ZERGFLYERARMORSLEVEL1) and greaterspire.is_idle and AbilityId.RESEARCH_ZERGFLYERARMORLEVEL1 in abilities and self.already_pending(
                            UpgradeId.ZERGFLYERARMORSLEVEL1) == 0:
                            greaterspire.research(UpgradeId.ZERGFLYERARMORSLEVEL1)

                            print('research zerg fly armor plus 1 ')
                        if self.structures(UnitTypeId.GREATERSPIRE).ready and self.already_pending(
                                UpgradeId.ZERGFLYERARMORSLEVEL2) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGFLYERARMORSLEVEL2) and greaterspire.is_idle and AbilityId.RESEARCH_ZERGFLYERARMORLEVEL2 in abilities:
                                greaterspire.research(UpgradeId.ZERGFLYERARMORSLEVEL2)
                                print('research zerg fly armor plus 2 ')
                        if self.structures(UnitTypeId.GREATERSPIRE).ready and self.already_pending(
                                UpgradeId.ZERGFLYERARMORSLEVEL1) == 0:
                            if self.can_afford(
                                    UpgradeId.ZERGFLYERARMORSLEVEL3) and greaterspire.is_idle and AbilityId.RESEARCH_ZERGFLYERARMORLEVEL3 in abilities:
                                greaterspire.research(UpgradeId.ZERGFLYERARMORSLEVEL3)
                                print('research zerg fly armor plus 3 ')


        elif action == 17:
            print(f'action={action}')
            if self.units(UnitTypeId.OVERLORD).exists and self.units(UnitTypeId.DRONE).exists and self.structures(
                    UnitTypeId.HIVE).exists and self.structures(
                UnitTypeId.ULTRALISKCAVERN).amount + self.already_pending(UnitTypeId.ULTRALISKCAVERN) == 0:
                building_place = self.townhalls.random.position
                placement_position = await self.find_placement(UnitTypeId.ULTRALISKCAVERN,
                                                               near=building_place,
                                                               placement_step=4)
                if placement_position is not None:
                    await self.build(UnitTypeId.ULTRALISKCAVERN, near=placement_position)
                    print('build ultraliskcavern')
            if self.units(UnitTypeId.ULTRALISKCAVERN).exists:
                ultraliskcaverns = self.structures(UnitTypeId.ULTRALISKCAVERN)
                abilities = await self.get_available_abilities(ultraliskcaverns)
                for ultraliskcavern in ultraliskcaverns:
                    if ultraliskcavern.is_idle:
                        if self.structures(UnitTypeId.ULTRALISKCAVERN).ready and self.can_afford(
                                UpgradeId.CHITINOUSPLATING) and ultraliskcavern.is_idle and AbilityId.RESEARCH_CHITINOUSPLATING in abilities and self.already_pending(
                            UpgradeId.CHITINOUSPLATING) == 0:
                            ultraliskcavern.research(UpgradeId.CHITINOUSPLATING)

                            print('research chitionous splating ')
                        if self.structures(UnitTypeId.ULTRALISKCAVERN).ready and self.can_afford(
                                UpgradeId.ANABOLICSYNTHESIS) and ultraliskcavern.is_idle and AbilityId.RESEARCH_ANABOLICSYNTHESIS in abilities and self.already_pending(
                            UpgradeId.ANABOLICSYNTHESIS) == 0:
                            ultraliskcavern.research(UpgradeId.ANABOLICSYNTHESIS)
                            print('research anabolic synthesis ')



        elif action == 18:
            print(f'action={action}')
            if self.units(UnitTypeId.OVERLORD).exists and self.units(UnitTypeId.DRONE).exists and self.structures(
                    UnitTypeId.HIVE).exists and self.structures(UnitTypeId.HYDRALISKDEN).exists and self.structures(
                UnitTypeId.LURKERDEN).amount + self.already_pending(UnitTypeId.LURKERDEN) == 0:
                building_place = self.townhalls.random.position
                placement_position = await self.find_placement(UnitTypeId.LURKERDEN,
                                                               near=building_place,
                                                               placement_step=4)
                if placement_position is not None:
                    await self.build(UnitTypeId.LURKERDEN, near=placement_position)
                    print('build lurkerden')
            if self.structures(UnitTypeId.LURKERDEN).exists:
                lurkerdens = self.structures(UnitTypeId.LURKERDEN)
                abilities = await self.get_available_abilities(lurkerdens)
                for lurkerden in lurkerdens:
                    if lurkerden.is_idle:
                        if self.structures(UnitTypeId.LURKERDEN).ready and self.can_afford(
                                UpgradeId.LURKERRANGE) and lurkerden.is_idle and AbilityId.LURKERDENRESEARCH_RESEARCHLURKERRANGE in abilities and self.already_pending(
                            UpgradeId.LURKERRANGE) == 0:
                            lurkerden.research(UpgradeId.LURKERRANGE)

                            print('research  lurker range')
                        if self.structures(UnitTypeId.LURKERDEN).ready and self.can_afford(
                                UpgradeId.DIGGINGCLAWS) and lurkerden.is_idle and AbilityId.RESEARCH_ADAPTIVETALONS in abilities and self.already_pending(
                            UpgradeId.DIGGINGCLAWS) == 0:
                            lurkerden.research(UpgradeId.DIGGINGCLAWS)
                            print('research diggingclaws ')
        elif action == 19:
            print(f'action={action}')
            if self.townhalls.exists and self.units(UnitTypeId.DRONE).exists and self.structures(
                    UnitTypeId.SPAWNINGPOOL).exists:
                bases = self.townhalls
                for base in bases:
                    if self.units(UnitTypeId.QUEEN).amount + self.already_pending(UnitTypeId.QUEEN) <= 10:
                        if base.is_idle and self.can_afford(UnitTypeId.QUEEN) and self.supply_left >= 2:
                            base.train(UnitTypeId.QUEEN)
                            print('train queen')
                            break

        elif action == 20:
            print(f'action={action}')
            larvae: Units = self.larva

            if self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 1 and larvae:
                    if self.can_afford(UnitTypeId.ZERGLING):
                        larvae.random.train(UnitTypeId.ZERGLING)
                        print('train zergling')
        elif action == 21:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.ROACHWARREN).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 2 and larvae:
                    if self.can_afford(UnitTypeId.ROACH):
                        larvae.random.train(UnitTypeId.ROACH)
                        print('train roach')
        elif action == 22:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.HYDRALISKDEN).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 2 and larvae:
                    if self.can_afford(UnitTypeId.HYDRALISK):
                        larvae.random.train(UnitTypeId.HYDRALISK)
                        print('train hydralisk')
        elif action == 23:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.BANELINGNEST).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists and self.units(UnitTypeId.ZERGLING).exists:
                zerglings = self.units(UnitTypeId.ZERGLING)

                for ling in zerglings:
                    if ling.is_idle and self.can_afford(
                            UnitTypeId.BANELING) and self.structures(UnitTypeId.BANELINGNEST).exists :
                        ling.build(UnitTypeId.BANELING)
                        print('morph baneling')
                        break
        elif action == 24:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.ROACHWARREN).exists and self.townhalls.exists and self.units(
                    UnitTypeId.ROACH).exists:
                roaches = self.units(UnitTypeId.ROACH)
                abilities = await self.get_available_abilities(roaches)

                for roach in roaches:
                    if roach.is_idle and self.can_afford(
                            UnitTypeId.RAVAGER) and AbilityId.MORPHTORAVAGER_RAVAGER in abilities and self.supply_left>=1:
                        roach.build(UnitTypeId.RAVAGER)
                        print('morph RAVAGER')
                        break

        elif action == 25:
            print(f'action={action}')

            try:
                self.last_sent
            except:
                self.last_sent = 0

            if (iteration - self.last_sent) > 200:
                if self.units(UnitTypeId.ZERGLING).exists:
                    if self.units(UnitTypeId.ZERGLING).idle.exists:
                        zergling = random.choice(self.units(UnitTypeId.ZERGLING).idle)
                    else:
                        zergling = random.choice(self.units(UnitTypeId.ZERGLING))
                    zergling.attack(self.enemy_start_locations[0])
                    self.last_sent = iteration
                    print('zergling scouting')
        elif action == 26:
            print(f'action={action}')

            try:
                self.last_sent
            except:
                self.last_sent = 0

            if (iteration - self.last_sent) > 100:
                if self.units(UnitTypeId.OVERLORD).exists:
                    if self.units(UnitTypeId.OVERLORD).idle.exists:
                        overlord = random.choice(self.units(UnitTypeId.OVERLORD).idle)
                    else:
                        overlord = random.choice(self.units(UnitTypeId.OVERLORD))
                    overlord.attack(self.enemy_start_locations[0])
                    self.last_sent = iteration
                    print('overlord scouting')
        elif action == 27:
            print(f'action={action}')

            try:
                self.last_sent
            except:
                self.last_sent = 0

            if (iteration - self.last_sent) > 300:
                if self.units(UnitTypeId.OVERSEER).exists:
                    if self.units(UnitTypeId.OVERSEER).idle.exists:
                        overlord = random.choice(self.units(UnitTypeId.OVERSEER).idle)
                    else:
                        overlord = random.choice(self.units(UnitTypeId.OVERSEER))
                    overlord.attack(self.enemy_start_locations[0])
                    self.last_sent = iteration
                    print('overseer scouting')
        elif action == 28:
            print(f'action={action}')
            if self.structures(UnitTypeId.LURKERDEN).exists and self.townhalls.exists and self.units(
                    UnitTypeId.HYDRALISK).exists:
                hydras = self.units(UnitTypeId.HYDRALISK)
                abilities = await self.get_available_abilities(hydras)

                for hydra in hydras:
                    if hydra.is_idle and self.can_afford(
                            UnitTypeId.LURKER) and AbilityId.MORPH_LURKER in abilities and self.supply_left>=1:
                        hydra.build(UnitTypeId.LURKER)
                        print('morph lurker')
                        break

        elif action == 29:
            print(f'action={action}')
            '''
            try:
                self.last_attack
            except:
                self.last_attack = 0
            '''
            if self.supply_army > 0:
                for unit in self.units.of_type(
                        {UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.QUEEN, UnitTypeId.ROACH,
                         UnitTypeId.OVERSEER,
                         UnitTypeId.ULTRALISK, UnitTypeId.MUTALISK, UnitTypeId.INFESTOR, UnitTypeId.CORRUPTOR,
                         UnitTypeId.BROODLORD,
                         UnitTypeId.OVERSEER, UnitTypeId.RAVAGER, UnitTypeId.VIPER, UnitTypeId.SWARMHOSTMP,
                         UnitTypeId.LURKER}):
                    try:
                        if self.enemy_units.closer_than(30, unit):
                            unit.attack(random.choice(unit.closer_than(30, unit)))
                            reward += 0.015
                            print('attack')
                        elif self.enemy_structures.closer_than(30, unit):
                            unit.attack(random.choice(self.enemy_structures.closer_than(30, unit)))
                            reward += 0.015
                            print('attack')
                        if self.units(UnitTypeId.VOIDRAY).amount > 6:
                            if self.enemy_units:
                                unit.attack(random.choice(self.enemy_units))
                                reward += 0.005
                                print('attack')
                            elif self.enemy_structures:
                                unit.attack(random.choice(self.enemy_structures))
                                reward += 0.005
                                print('attack')
                            elif self.enemy_start_locations:
                                unit.attack(self.enemy_start_locations[0])
                                print('attack')
                                reward += 0.005
                            self.last_attack = iteration

                    except Exception as e:
                        print(e)
                        pass

        elif action == 30:
            if self.supply_army > 0:
                print(f'action={action}')
                for unit in self.units.of_type(
                        {UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.QUEEN, UnitTypeId.ROACH,
                         UnitTypeId.OVERSEER,
                         UnitTypeId.ULTRALISK, UnitTypeId.MUTALISK, UnitTypeId.INFESTOR, UnitTypeId.CORRUPTOR,
                         UnitTypeId.BROODLORD,
                         UnitTypeId.OVERSEER, UnitTypeId.RAVAGER, UnitTypeId.VIPER, UnitTypeId.SWARMHOSTMP,
                         UnitTypeId.LURKER}):
                    try:
                        '''
                        if iteration - self.last_attack < 200:
                            pass
                        else:
                        '''
                        where2retreat = random.choice(self.townhalls)
                        print(self.start_location)
                        unit.move(where2retreat)
                        pass
                        print('retreat')
                    except Exception as e:
                        print(e)
                        pass
        elif action == 31:
            print(f'action={action}')
            if self.townhalls.exists and self.units(UnitTypeId.QUEEN).exists:
                hq = self.townhalls.random
                queens = self.units(UnitTypeId.QUEEN)
                # Send idle queens with >=25 energy to inject
                for queen in self.units(UnitTypeId.QUEEN).idle:
                    # The following checks if the inject ability is in the queen abilitys - basically it checks if we have enough energy and if the ability is off-cooldown
                    abilities = await self.get_available_abilities(queens)
                    if AbilityId.EFFECT_INJECTLARVA in abilities:
                        if queen.energy >= 25:
                            hq = self.townhalls.closest_to(queen)
                            queen(AbilityId.EFFECT_INJECTLARVA, hq)
                            print('queen inject')
                            break
        elif action == 32:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.SPIRE).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 2 and larvae:
                    if self.can_afford(UnitTypeId.MUTALISK):
                        larvae.random.train(UnitTypeId.MUTALISK)
                        print('train mutalisk')

        elif action == 31:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.SPIRE).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 2 and larvae:
                    if self.can_afford(UnitTypeId.CORRUPTOR):
                        larvae.random.train(UnitTypeId.CORRUPTOR)
                        print('train corruptor')
        elif action == 32:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.INFESTATIONPIT).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 2 and larvae:
                    if self.can_afford(UnitTypeId.INFESTOR) and self.units(UnitTypeId.INFESTOR).amount+self.already_pending(UnitTypeId.INFESTOR)<=3:
                        larvae.random.train(UnitTypeId.INFESTOR)
                        print('train infestor')
        elif action == 33:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.ULTRALISKCAVERN).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 6 and larvae:
                    if self.can_afford(UnitTypeId.ULTRALISK):
                        larvae.random.train(UnitTypeId.ULTRALISK)
                        print('train ultralisk')
        elif action == 34:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.HIVE).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 3 and larvae and self.units(UnitTypeId.VIPER).amount+self.already_pending(UnitTypeId.VIPER)<=2:
                    if self.can_afford(UnitTypeId.VIPER):
                        larvae.random.train(UnitTypeId.VIPER)
                        print('train viper')
        elif action == 35:
            print(f'action={action}')
            if self.structures(UnitTypeId.GREATERSPIRE).exists and self.townhalls.exists and self.units(
                    UnitTypeId.CORRUPTOR).exists:
                corruptors = self.units(UnitTypeId.CORRUPTOR)
                abilities = await self.get_available_abilities(corruptors)

                for corruptor in corruptors:
                    if corruptor.is_idle and self.can_afford(
                            UnitTypeId.BROODLORD) and AbilityId.MORPHTOBROODLORD_BROODLORD in abilities and self.supply_left>=2:
                        corruptor.build(UnitTypeId.BROODLORD)
                        print('morph broodlord')
                        break
        elif action == 36:
            print(f'action={action}')
            if self.structures(UnitTypeId.LAIR).exists or self.structures(UnitTypeId.HIVE).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                overlords = self.units(UnitTypeId.OVERLORD)
                abilities = await self.get_available_abilities(overlords)

                for overlord in overlords:
                    if overlord.is_idle and self.can_afford(
                            UnitTypeId.OVERSEER) and AbilityId.MORPH_OVERSEER in abilities and self.units(UnitTypeId.OVERSEER).amount+self.already_pending(UnitTypeId.OVERSEER)<=3:
                        overlord.build(UnitTypeId.OVERSEER)
                        print('morph overseer')
                        break
        elif action == 37:
            print(f'action={action}')
            larvae: Units = self.larva
            if self.structures(UnitTypeId.INFESTATIONPIT).exists and self.townhalls.exists and self.units(
                    UnitTypeId.OVERLORD).exists:
                if self.supply_left >= 4 and larvae:
                    if self.can_afford(UnitTypeId.SWARMHOSTMP) and self.units(UnitTypeId.SWARMHOSTMP).amount+self.already_pending(UnitTypeId.SWARMHOSTMP)<=2:
                        larvae.random.train(UnitTypeId.SWARMHOSTMP)
                        print('train swarm host')

        elif action == 38:
            print(f'action={action}')
            if self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.townhalls.exists:
                if 3<=self.townhalls.amount <= 4:
                    nexus = self.townhalls.random

                    if self.can_afford(UnitTypeId.SPORECRAWLER) + self.structures(
                            UnitTypeId.SPORECRAWLER).amount + self.already_pending(UnitTypeId.SPORECRAWLER) <= 2:
                        place_position = nexus.position
                        placement_position = await self.find_placement(UnitTypeId.SPORECRAWLER,
                                                                       near=place_position,
                                                                       placement_step=3)
                        if placement_position is not None:
                            await self.build(UnitTypeId.SPORECRAWLER, near=placement_position)
                            print('build spore crawler')
                elif 5<=self.townhalls.amount :
                    nexus = self.townhalls.random

                    if self.can_afford(UnitTypeId.SPORECRAWLER) + self.structures(
                            UnitTypeId.SPORECRAWLER).amount + self.already_pending(UnitTypeId.SPORECRAWLER) <= 4:
                        place_position = nexus.position
                        placement_position = await self.find_placement(UnitTypeId.SPORECRAWLER,
                                                                       near=place_position,
                                                                       placement_step=3)
                        if placement_position is not None:
                            await self.build(UnitTypeId.SPORECRAWLER, near=placement_position)
                            print('build spore crawler')

        elif action == 39:
            print(f'action={action}')
            if self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.townhalls.exists:
                if 3<=self.townhalls.amount <= 4:
                    base = self.townhalls.random

                    if self.can_afford(UnitTypeId.SPINECRAWLER) and self.structures(
                            UnitTypeId.SPINECRAWLER).amount + self.already_pending(UnitTypeId.SPINECRAWLER) <= 1:
                        place_position = base.position
                        placement_position = await self.find_placement(UnitTypeId.SPINECRAWLER,
                                                                       near=place_position,
                                                                       placement_step=3)
                        if placement_position is not None:
                            await self.build(UnitTypeId.SPINECRAWLER, near=placement_position)
                            print('build spine crawler')
                elif 5<=self.townhalls.amount :
                    base = self.townhalls.random

                    if self.can_afford(UnitTypeId.SPINECRAWLER) and self.structures(
                            UnitTypeId.SPINECRAWLER).amount + self.already_pending(UnitTypeId.SPINECRAWLER) <= 3:
                        place_position = base.position
                        placement_position = await self.find_placement(UnitTypeId.SPINECRAWLER,
                                                                       near=place_position,
                                                                       placement_step=3)
                        if placement_position is not None:
                            await self.build(UnitTypeId.SPINECRAWLER, near=placement_position)
                            print('build spine crawler')
        # do nothing
        elif action == 40:
            print(f'action={action}')

        self.lock.acquire()
        self.transaction['action'] = None
        self.transaction['isTTO'] = True
        self.transaction['reward'] = reward
        self.transaction['iter'] = iteration
        self.lock.release()
        '''
        while True:
            try:
                with open('transaction.pkl','wb') as f:
                    pickle.dump(transaction,f)
                break
            except Exception as e:
                time.sleep(0.001)
                pass
        '''

    async def defend(self):
        if self.structures(UnitTypeId.NEXUS).exists:
            for nexus in self.townhalls:
                if self.enemy_units.amount >= 2 and self.enemy_units.closest_distance_to(nexus) < 30:
                    self.rally_defend = True
                    for unit in self.units.of_type(
                            {UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.QUEEN, UnitTypeId.ROACH,
                             UnitTypeId.OVERSEER,
                             UnitTypeId.ULTRALISK, UnitTypeId.MUTALISK, UnitTypeId.INFESTOR, UnitTypeId.CORRUPTOR,
                             UnitTypeId.BROODLORD,
                             UnitTypeId.OVERSEER, UnitTypeId.RAVAGER, UnitTypeId.VIPER, UnitTypeId.SWARMHOSTMP,
                             UnitTypeId.LURKER}):
                        closed_enemy = self.enemy_units.sorted(lambda x: x.distance_to(unit))
                        unit.attack(closed_enemy[0])
                else:
                    self.rally_defend = False

            if self.rally_defend == True:
                map_center = self.game_info.map_center
                rally_point = self.townhalls.random.position.towards(map_center, distance=5)
                for unit in self.units.of_type(
                        {UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.QUEEN, UnitTypeId.ROACH,
                         UnitTypeId.OVERSEER,
                         UnitTypeId.ULTRALISK, UnitTypeId.MUTALISK, UnitTypeId.INFESTOR, UnitTypeId.CORRUPTOR,
                         UnitTypeId.BROODLORD,
                         UnitTypeId.OVERSEER, UnitTypeId.RAVAGER, UnitTypeId.VIPER, UnitTypeId.SWARMHOSTMP,
                         UnitTypeId.LURKER}):
                    if unit.distance_to(self.start_location) > 100 and unit not in self.unit_tags_received_action:
                        unit.move(rally_point)


def worker(transaction, lock):
    laddermap_2023 = ['Altitude LE', 'Ancient Cistern LE', 'Babylon LE', 'Dragon Scales LE', 'Gresvan LE',
                      'Neohumanity LE', 'Royal Blood LE']
    res = run_game(maps.get(laddermap_2023[0]),
                   [Bot(Race.Zerg, Zerg_bot(transaction, lock)), Computer(Race.Terran, Difficulty.Easy)],
                   realtime=False)

    lock.acquire()
    transaction['done'] = True
    transaction['res'] = res
    lock.release()
    '''
    while True:
        try:
            with open('transaction.pkl', 'rb') as f:
                transaction = pickle.load(f)
                break
            time.sleep(0.001)
        except Exception as e:
            time.sleep(0.001)
            pass
    transaction['done'] = True
    transaction['res'] = res
    while True:
        try:
            with open('transaction.pkl','wb') as f:
                pickle.dump(transaction,f)
            break
        except Exception as e:
            time.sleep(0.001)
            pass

    '''
