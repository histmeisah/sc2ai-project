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


class WorkerRunshBot(BotAI):
    def __init__(self, transaction, lock):
        self.lock = lock
        self.transaction = transaction
        self.worker_supply = 12  # 农民数量
        self.army_supply = 0  # 部队人口
        self.base_count = 1  # 基地数量
        self.enemy_units_count = 0
        # self.army_units_list=[]  # 我方部队列表
        # self.enemy_list = []
        self.by_count = 0
        self.bf_count = 0
        self.vs_count = 0
        self.vr_count = 0
        self.vc_count = 0
        self.vf_count = 0
        self.vb_count = 0

        self.base_pending = 0
        self.gateway_count = 0
        self.rally_defend = False

    def get_information(self):
        self.worker_supply = self.workers.amount
        self.army_supply = self.supply_army
        self.base_count = self.structures(UnitTypeId.NEXUS).amount
        self.base_pending = self.already_pending(UnitTypeId.NEXUS)
        self.by_count = self.structures(UnitTypeId.CYBERNETICSCORE).amount
        self.bf_count = self.structures(UnitTypeId.FORGE).amount
        self.vc_count = self.structures(UnitTypeId.TWILIGHTCOUNCIL).amount
        self.vs_count = self.structures(UnitTypeId.STARGATE).amount + self.already_pending(UnitTypeId.STARGATE)
        self.vf_count = self.structures(UnitTypeId.FLEETBEACON).amount
        self.vr_count = self.structures(UnitTypeId.ROBOTICSFACILITY).amount + self.already_pending(
            UnitTypeId.ROBOTICSFACILITY)
        self.vb_count = self.structures(UnitTypeId.ROBOTICSBAY).amount
        # self.get_army_list()
        self.enemy_units_count = self.enemy_units.amount
        self.gateway_count = self.structures(UnitTypeId.GATEWAY).amount

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
                'by_count': self.by_count,
                'bf_count': self.bf_count,
                'vs_count': self.vs_count,  # 我方vs数量
                'vr_count': self.vr_count,
                'base_pending': self.base_pending,  # 我方正在建造的基地数量
                'gateway_count': self.gateway_count,  # bg_(不能折跃)数量
                'warp_gate_count': self.warp_gate_count}  # 折跃门数量

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
        if self.structures(UnitTypeId.NEXUS).exists and self.supply_army >= 2:
            for nexus in self.townhalls:
                if self.enemy_units.amount >= 2 and self.enemy_units.closest_distance_to(nexus) < 30:
                    self.rally_defend = True
                    for unit in self.units.of_type(
                            {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER, UnitTypeId.SENTRY,
                             UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
                             UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
                             UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
                             UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
                             UnitTypeId.CHANGELINGZEALOT}):
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
            print(f'action={action}')
            print('建设水晶塔')
            if self.structures(UnitTypeId.NEXUS).exists and self.units(UnitTypeId.PROBE).exists:
                if '00:00' <= self.time_formatted <= '06:00':
                    if self.supply_left <= 3 and self.already_pending(
                            UnitTypeId.PYLON) <= 2 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.PYLON):
                            base = self.townhalls.random
                            place_position = base.position + Point2((0, self.Location * 8))
                            await self.build(UnitTypeId.PYLON, near=place_position, placement_step=2)
                if '06:00' <= self.time_formatted <= '07:00':
                    if self.supply_left <= 5 and self.already_pending(
                            UnitTypeId.PYLON) <= 4 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.PYLON):
                            base = self.townhalls.random

                            place_position = base.position + Point2((0, self.Location * 8))
                            await self.build(UnitTypeId.PYLON, near=place_position, placement_step=2)
                if '06:00' <= self.time_formatted <= '08:00':
                    if self.supply_left <= 5 and self.already_pending(
                            UnitTypeId.PYLON) <= 3 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.PYLON):
                            base = self.townhalls.random

                            place_position = base.position
                            await self.build(UnitTypeId.PYLON, near=place_position, placement_step=2)
                if '08:00' <= self.time_formatted <= '10:00':
                    if self.supply_left <= 7 and self.already_pending(
                            UnitTypeId.PYLON) <= 4 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.PYLON):
                            base = self.townhalls.random

                            place_position = base.position
                            await self.build(UnitTypeId.PYLON, near=place_position, placement_step=2)
                if '10:00' <= self.time_formatted:
                    if self.supply_left <= 7 and self.already_pending(
                            UnitTypeId.PYLON) <= 4 and not self.supply_cap == 200:
                        if self.can_afford(UnitTypeId.PYLON):
                            base = self.townhalls.random
                            place_position = base.position + Point2((0, self.Location * 8))
                            await self.build(UnitTypeId.PYLON, near=place_position, placement_step=2)
        elif action == 1:
            print(f'action={action}')
            if self.structures(UnitTypeId.NEXUS).exists:
                for nexus in self.townhalls:
                    # 探机
                    if self.workers.amount + self.already_pending(UnitTypeId.PROBE) <= 70 and self.supply_left > 0:
                        if self.can_afford(UnitTypeId.PROBE) and nexus.is_idle:
                            nexus.train(UnitTypeId.PROBE)
                            print('训练探机')
        elif action == 2:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists:
                # 吸收间
                for nexus in self.townhalls:
                    for vespene in self.vespene_geyser.closer_than(10, nexus):
                        if self.can_afford(UnitTypeId.ASSIMILATOR) and not self.structures(
                                UnitTypeId.ASSIMILATOR).closer_than(2, vespene):
                            await self.build(UnitTypeId.ASSIMILATOR, vespene)
                            have_builded = True
                            print('建设气矿')
        elif action == 3:
            if self.units(UnitTypeId.PROBE).exists:

                print(f'action={action}')
                if self.can_afford(UnitTypeId.NEXUS) and self.already_pending(
                        UnitTypeId.NEXUS) == 0 and self.structures(UnitTypeId.NEXUS).amount <= 6:
                    await self.expand_now()
                    print('扩建基地')
        elif action == 4:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists:
                for nexus in self.townhalls:
                    if not self.structures(UnitTypeId.GATEWAY).closer_than(10, nexus).exists:
                        if self.can_afford(UnitTypeId.GATEWAY) and self.already_pending(UnitTypeId.GATEWAY) == 0:
                            building_place = self.structures(UnitTypeId.PYLON).closest_to(nexus).position
                            placement_position = await self.find_placement(UnitTypeId.GATEWAY, near=building_place,
                                                                           placement_step=2)
                            if placement_position is not None:
                                await self.build(UnitTypeId.GATEWAY, near=placement_position)
                                print('建设BG')
        elif action == 5:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists:
                if self.structures(UnitTypeId.GATEWAY).exists:
                    for nexus in self.townhalls:
                        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists and self.already_pending(
                                UnitTypeId.CYBERNETICSCORE) == 0:
                            if self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.already_pending(
                                    UnitTypeId.CYBERNETICSCORE) == 0 and not self.structures(
                                UnitTypeId.CYBERNETICSCORE).exists:
                                building_place = self.structures(UnitTypeId.PYLON).closest_to(nexus).position
                                placement_position = await self.find_placement(UnitTypeId.CYBERNETICSCORE,
                                                                               near=building_place,
                                                                               placement_step=4)
                                if placement_position is not None:
                                    await self.build(UnitTypeId.CYBERNETICSCORE, near=placement_position)
                                    print('建设BY')

            if self.structures(UnitTypeId.CYBERNETICSCORE).exists:
                by = self.structures(UnitTypeId.CYBERNETICSCORE).random
                abilities = await self.get_available_abilities(by)

                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UpgradeId.WARPGATERESEARCH) == 0:
                    if self.can_afford(
                            UpgradeId.WARPGATERESEARCH) and by.is_idle and AbilityId.RESEARCH_WARPGATE in abilities:
                        by.research(UpgradeId.WARPGATERESEARCH)
                        print('研究折跃门')
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UpgradeId.PROTOSSAIRWEAPONSLEVEL1) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSAIRWEAPONSLEVEL1) and by.is_idle and AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1 in abilities:
                        by.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL1)
                        print('研究空军1攻')
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UpgradeId.PROTOSSAIRWEAPONSLEVEL2) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSAIRWEAPONSLEVEL2) and by.is_idle and AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2 in abilities:
                        by.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL2)
                        print('研究空军2攻')
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UpgradeId.PROTOSSAIRWEAPONSLEVEL3) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSAIRWEAPONSLEVEL3) and by.is_idle and AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3 in abilities:
                        by.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL3)
                        print('研究空军3攻')
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UpgradeId.PROTOSSAIRARMORSLEVEL1) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSAIRARMORSLEVEL1) and by.is_idle and AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1 in abilities:
                        by.research(UpgradeId.PROTOSSAIRARMORSLEVEL1)
                        print('研究空军1防')
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UpgradeId.PROTOSSAIRARMORSLEVEL2) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSAIRARMORSLEVEL2) and by.is_idle and AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2 in abilities:
                        by.research(UpgradeId.PROTOSSAIRARMORSLEVEL2)
                        print('研究空军2防')
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UpgradeId.PROTOSSAIRARMORSLEVEL3) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSAIRARMORSLEVEL3) and by.is_idle and AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3 in abilities:
                        by.research(UpgradeId.PROTOSSAIRARMORSLEVEL3)
                        print('研究空军3防')
        elif action == 6:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists and self.structures(UnitTypeId.CYBERNETICSCORE).exists:

                for nexus in self.townhalls:
                    if not self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists and self.already_pending(
                            UnitTypeId.TWILIGHTCOUNCIL) == 0:
                        if self.can_afford(UnitTypeId.TWILIGHTCOUNCIL) and self.already_pending(
                                UnitTypeId.TWILIGHTCOUNCIL) == 0 and self.structures(UnitTypeId.CYBERNETICSCORE).exists:
                            place_position = nexus.position
                            placement_position = await self.find_placement(UnitTypeId.TWILIGHTCOUNCIL,
                                                                           near=place_position.towards(
                                                                               self.game_info.map_center, 6),
                                                                           placement_step=2)
                            if placement_position is not None:
                                await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=placement_position)
                                print('建设VC')
        elif action == 7:
            print(f'action={action}')
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
                vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).random
                abilities = await self.get_available_abilities(vc)
                if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready and self.already_pending(
                        UpgradeId.ADEPTPIERCINGATTACK) == 0:
                    if self.can_afford(
                            UpgradeId.ADEPTPIERCINGATTACK) and vc.is_idle and AbilityId.RESEARCH_ADEPTRESONATINGGLAIVES in abilities:
                        vc.research(UpgradeId.ADEPTPIERCINGATTACK)
                        print('研究使徒攻速')

        elif action == 8:
            print(f'action={action}')
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
                vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).random
                abilities = await self.get_available_abilities(vc)
                if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready and self.already_pending(UpgradeId.BLINKTECH) == 0:
                    if self.can_afford(UpgradeId.BLINKTECH) and vc.is_idle and AbilityId.RESEARCH_BLINK in abilities:
                        vc.research(UpgradeId.BLINKTECH)
                        print('研究闪烁追猎')
        elif action == 9:
            print(f'action={action}')
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
                vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).random
                abilities = await self.get_available_abilities(vc)
                if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready and self.already_pending(UpgradeId.CHARGE) == 0:
                    if self.can_afford(UpgradeId.CHARGE) and vc.is_idle and AbilityId.RESEARCH_CHARGE in abilities:
                        vc.research(UpgradeId.CHARGE)
                        print('研究冲锋狂热者')


        elif action == 10:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists:
                if not self.structures(UnitTypeId.FORGE).amount < 2 and self.already_pending(
                        UnitTypeId.FORGE) + self.structures(UnitTypeId.FORGE).amount < 2:
                    if self.can_afford(UnitTypeId.FORGE) and self.already_pending(
                            UnitTypeId.FORGE) <= 2:
                        building_place = self.structures(UnitTypeId.PYLON).random.position
                        placement_position = await self.find_placement(UnitTypeId.FORGE,
                                                                       near=building_place,
                                                                       placement_step=2)
                        if placement_position is not None:
                            await self.build(UnitTypeId.FORGE, near=placement_position)
                            print('建设BF')
            if self.structures(UnitTypeId.FORGE).exists:
                bf = self.structures(UnitTypeId.FORGE).random
                abilities = await self.get_available_abilities(bf)
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) == 0:
                    if self.can_afford(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) \
                            and bf.is_idle and \
                            AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1 in abilities:
                        bf.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1)
                        print('研究地面1攻')
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) and bf.is_idle and AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2 in abilities:
                        bf.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2)
                        print('研究地面2攻')
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) and bf.is_idle and AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3 in abilities:
                        bf.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3)
                        print('研究地面3攻')
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSGROUNDARMORSLEVEL1) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSGROUNDARMORSLEVEL1) and bf.is_idle and AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1 in abilities:
                        bf.research(UpgradeId.PROTOSSGROUNDARMORSLEVEL1)
                        print('研究地面1防')
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSGROUNDARMORSLEVEL2) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSGROUNDARMORSLEVEL2) and bf.is_idle and AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL2 in abilities:
                        bf.research(UpgradeId.PROTOSSGROUNDARMORSLEVEL2)
                        print('研究地面2防')
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSGROUNDARMORSLEVEL3) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSGROUNDARMORSLEVEL3) and bf.is_idle and AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3 in abilities:
                        bf.research(UpgradeId.PROTOSSGROUNDARMORSLEVEL3)
                        print('研究地面3防')
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSSHIELDSLEVEL1) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSSHIELDSLEVEL1) and bf.is_idle and AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1 in abilities:
                        bf.research(UpgradeId.PROTOSSSHIELDSLEVEL1)
                        print('研究1盾')
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSSHIELDSLEVEL2) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSSHIELDSLEVEL2) and bf.is_idle and AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2 in abilities:
                        bf.research(UpgradeId.PROTOSSSHIELDSLEVEL2)
                        print('研究2盾')
                if self.structures(UnitTypeId.FORGE).ready and self.already_pending(
                        UpgradeId.PROTOSSSHIELDSLEVEL3) == 0:
                    if self.can_afford(
                            UpgradeId.PROTOSSSHIELDSLEVEL3) and bf.is_idle and AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3 in abilities:
                        bf.research(UpgradeId.PROTOSSSHIELDSLEVEL3)
                        print('研究3盾')
        elif action == 11:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists and self.structures(UnitTypeId.ROBOTICSFACILITY).exists:
                if not self.structures(UnitTypeId.ROBOTICSBAY).exists and self.already_pending(
                        UnitTypeId.ROBOTICSBAY) == 0:
                    if self.can_afford(UnitTypeId.ROBOTICSBAY) and self.already_pending(
                            UnitTypeId.ROBOTICSBAY) == 0 and self.structures(UnitTypeId.ROBOTICSFACILITY).exists:
                        building_place = self.structures(UnitTypeId.PYLON).random.position
                        placement_position = await self.find_placement(UnitTypeId.ROBOTICSBAY,
                                                                       near=building_place,
                                                                       placement_step=2)
                        if placement_position is not None:
                            await self.build(UnitTypeId.ROBOTICSBAY, near=placement_position)
                            print('建设VB')
                if self.structures(UnitTypeId.ROBOTICSBAY).exists:
                    vb = self.structures(UnitTypeId.ROBOTICSBAY).random
                    abilities = await self.get_available_abilities(vb)
                    if self.structures(UnitTypeId.ROBOTICSBAY).ready and self.already_pending(
                            UpgradeId.EXTENDEDTHERMALLANCE) == 0:
                        if self.can_afford(
                                UpgradeId.EXTENDEDTHERMALLANCE) and vb.is_idle and AbilityId.RESEARCH_EXTENDEDTHERMALLANCE in abilities:
                            vb.research(UpgradeId.EXTENDEDTHERMALLANCE)
                            print('研究巨像射程')
                    if self.structures(UnitTypeId.ROBOTICSBAY).ready and self.already_pending(
                            UpgradeId.GRAVITICDRIVE) == 0:
                        if self.can_afford(
                                UpgradeId.GRAVITICDRIVE) and vb.is_idle and AbilityId.RESEARCH_GRAVITICDRIVE in abilities:
                            vb.research(UpgradeId.GRAVITICDRIVE)
                            print('研究棱镜速度')
                    if self.structures(UnitTypeId.ROBOTICSBAY).ready and self.already_pending(
                            UpgradeId.OBSERVERGRAVITICBOOSTER) == 0:
                        if self.can_afford(
                                UpgradeId.OBSERVERGRAVITICBOOSTER) and vb.is_idle and AbilityId.RESEARCH_GRAVITICBOOSTER in abilities:
                            vb.research(UpgradeId.OBSERVERGRAVITICBOOSTER)
                            print('研究OB速度')
        elif action == 12:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists and self.structures(UnitTypeId.CYBERNETICSCORE).exists:
                if self.can_afford(UnitTypeId.ROBOTICSFACILITY):
                    building_place = self.structures(UnitTypeId.PYLON).random.position
                    placement_position = await self.find_placement(UnitTypeId.ROBOTICSFACILITY,
                                                                   near=building_place,
                                                                   placement_step=2)
                    if placement_position is not None:
                        await self.build(UnitTypeId.ROBOTICSFACILITY, near=placement_position)
                        print('建设VR')
        elif action == 13:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists and self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
                if not self.structures(UnitTypeId.TEMPLARARCHIVE).exists and self.already_pending(
                        UnitTypeId.TEMPLARARCHIVE) == 0:
                    if self.can_afford(UnitTypeId.TEMPLARARCHIVE) and self.already_pending(
                            UnitTypeId.TEMPLARARCHIVE) == 0 and self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
                        building_place = self.structures(UnitTypeId.PYLON).random.position
                        placement_position = await self.find_placement(UnitTypeId.TEMPLARARCHIVE,
                                                                       near=building_place,
                                                                       placement_step=4)
                        if placement_position is not None:
                            await self.build(UnitTypeId.TEMPLARARCHIVE, near=placement_position)
                            print('建设VT')
            if self.structures(UnitTypeId.FLEETBEACON).exists:
                vt = self.structures(UnitTypeId.FLEETBEACON).random
                abilities = await self.get_available_abilities(vt)
                if self.structures(UnitTypeId.FLEETBEACON).ready and self.already_pending(UpgradeId.PSISTORMTECH) == 0:
                    if self.can_afford(
                            UpgradeId.PSISTORMTECH) and vt.is_idle and AbilityId.RESEARCH_PSISTORM in abilities:
                        vt.research(UpgradeId.PSISTORMTECH)
                        print('研究闪电')
        elif action == 14:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists and self.structures(UnitTypeId.STARGATE).exists:
                if not self.structures(UnitTypeId.FLEETBEACON).exists and self.already_pending(
                        UnitTypeId.FLEETBEACON) == 0:
                    if self.can_afford(UnitTypeId.FLEETBEACON) and self.already_pending(
                            UnitTypeId.FLEETBEACON) == 0 and self.structures(UnitTypeId.STARGATE).exists:
                        building_place = self.structures(UnitTypeId.PYLON).random.position
                        placement_position = await self.find_placement(UnitTypeId.FLEETBEACON,
                                                                       near=building_place,
                                                                       placement_step=4)
                        if placement_position is not None:
                            await self.build(UnitTypeId.FLEETBEACON, near=placement_position)
                            print('建设VF')
            if self.structures(UnitTypeId.FLEETBEACON).exists:
                vf = self.structures(UnitTypeId.FLEETBEACON).random
                abilities = await self.get_available_abilities(vf)
                if self.structures(UnitTypeId.FLEETBEACON).ready and self.already_pending(
                        UpgradeId.VOIDRAYSPEEDUPGRADE) == 0:
                    if self.can_afford(
                            UpgradeId.VOIDRAYSPEEDUPGRADE) and vf.is_idle and AbilityId.FLEETBEACONRESEARCH_RESEARCHVOIDRAYSPEEDUPGRADE in abilities:
                        vf.research(UpgradeId.VOIDRAYSPEEDUPGRADE)
                        print('研究虚空速度')
                if self.structures(UnitTypeId.FLEETBEACON).ready and self.already_pending(
                        UpgradeId.PHOENIXRANGEUPGRADE) == 0:
                    if self.can_afford(
                            UpgradeId.PHOENIXRANGEUPGRADE) and vf.is_idle and AbilityId.RESEARCH_PHOENIXANIONPULSECRYSTALS in abilities:
                        vf.research(UpgradeId.PHOENIXRANGEUPGRADE)
                        print('研究凤凰射程')
                if self.structures(UnitTypeId.FLEETBEACON).ready and self.already_pending(
                        UpgradeId.TEMPESTGROUNDATTACKUPGRADE) == 0:
                    if self.can_afford(
                            UpgradeId.TEMPESTGROUNDATTACKUPGRADE) and vf.is_idle and AbilityId.FLEETBEACONRESEARCH_TEMPESTRESEARCHGROUNDATTACKUPGRADE in abilities:
                        vf.research(UpgradeId.TEMPESTGROUNDATTACKUPGRADE)
                        print('研究风暴对建筑攻击')


        elif action == 15:
            print(f'action={action}')
            if self.structures(UnitTypeId.PYLON).exists and self.units(UnitTypeId.PROBE).exists and self.structures(
                    UnitTypeId.NEXUS).exists and self.structures(UnitTypeId.CYBERNETICSCORE).exists:
                for nexus in self.townhalls:
                    if self.can_afford(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=nexus)
                        reward += 0.015
                        print('建设VS')


        elif action == 16:
            print(f'action={action}')
            if self.structures(UnitTypeId.STARGATE).exists and self.supply_left >= 4:
                for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                    if self.can_afford(UnitTypeId.VOIDRAY):
                        sg.train(UnitTypeId.VOIDRAY)
                        reward += 0.015
                        print('训练光舰')
        elif action == 17:
            print(f'action={action}')
            if self.structures(UnitTypeId.STARGATE).exists and self.structures(UnitTypeId.FLEETBEACON).exists:

                for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                    if self.can_afford(UnitTypeId.CARRIER) and self.supply_left >= 6:
                        sg.train(UnitTypeId.CARRIER)
                        reward += 0.015
                        print('训练航母')
        elif action == 18:
            print(f'action={action}')
            if self.structures(UnitTypeId.STARGATE).exists and self.structures(UnitTypeId.FLEETBEACON).exists:
                for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                    if self.can_afford(UnitTypeId.TEMPEST) and self.supply_left >= 5:
                        sg.train(UnitTypeId.TEMPEST)
                        reward += 0.015
                        print('训练风暴战舰')
        elif action == 19:
            print(f'action={action}')
            if self.structures(UnitTypeId.STARGATE).exists:
                if self.units(UnitTypeId.ORACLE).amount + self.already_pending(UnitTypeId.ORACLE) <= 1:
                    for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                        if self.can_afford(UnitTypeId.ORACLE) and self.supply_left >= 3:
                            sg.train(UnitTypeId.ORACLE)
                            reward += 0.015
                            print('训练先知')
        elif action == 20:
            print(f'action={action}')
            if self.structures(UnitTypeId.STARGATE).exists:

                for sg in self.structures(UnitTypeId.STARGATE).ready.idle:
                    if self.can_afford(UnitTypeId.PHOENIX) and self.supply_left >= 2 and self.units(
                            UnitTypeId.PHOENIX).amount + self.already_pending(UnitTypeId.PHOENIX) <= 3:
                        sg.train(UnitTypeId.PHOENIX)
                        reward += 0.015
                        print('训练凤凰战机')
        elif action == 21:
            print(f'action={action}')
            if self.structures(UnitTypeId.NEXUS).exists:
                nexuses = self.structures(UnitTypeId.NEXUS)
                abilities = await self.get_available_abilities(nexuses)

                if self.structures(UnitTypeId.NEXUS).exists \
                        and self.structures(UnitTypeId.STARGATE).exists \
                        and self.structures(UnitTypeId.FLEETBEACON).exists:

                    for base in self.townhalls:
                        if self.can_afford(
                                UnitTypeId.MOTHERSHIP) and self.supply_left >= 10 and base.is_idle and AbilityId.NEXUSTRAINMOTHERSHIP_MOTHERSHIP in abilities:
                            base.train(UnitTypeId.MOTHERSHIP)
                            print('train mothership')

        elif action == 22:
            print(f'action={action}')

            try:
                self.last_sent
            except:
                self.last_sent = 0

            if (iteration - self.last_sent) > 200:
                if self.units(UnitTypeId.PROBE).exists:
                    if self.units(UnitTypeId.PROBE).idle.exists:
                        probe = random.choice(self.units(UnitTypeId.PROBE).idle)
                    else:
                        probe = random.choice(self.units(UnitTypeId.PROBE))
                    probe.attack(self.enemy_start_locations[0])
                    self.last_sent = iteration
                    print('侦查')
        elif action == 23:
            print(f'action={action}')
            if self.units(UnitTypeId.DARKTEMPLAR).exists:
                dts = self.units(UnitTypeId.DARKTEMPLAR)
                for dt in dts:
                    dt(AbilityId.MORPH_ARCHON)
                    print('合成执政官')

        elif action == 24:
            print(f'action={action}')
            '''
            try:
                self.last_attack
            except:
                self.last_attack = 0
            '''
            if self.supply_army > 0:
                for unit in self.units.of_type(
                        {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER,
                         UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
                         UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
                         UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
                         UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
                         UnitTypeId.CHANGELINGZEALOT}):
                    try:
                        if self.enemy_units.closer_than(30, unit):
                            unit.attack(random.choice(unit.closer_than(30, unit)))
                            reward += 0.015
                            print('进攻')
                        elif self.enemy_structures.closer_than(30, unit):
                            unit.attack(random.choice(self.enemy_structures.closer_than(30, unit)))
                            reward += 0.015
                            print('进攻')
                        if self.units(UnitTypeId.VOIDRAY).amount > 6:
                            if self.enemy_units:
                                unit.attack(random.choice(self.enemy_units))
                                reward += 0.005
                                print('进攻')
                            elif self.enemy_structures:
                                unit.attack(random.choice(self.enemy_structures))
                                reward += 0.005
                                print('进攻')
                            elif self.enemy_start_locations:
                                unit.attack(self.enemy_start_locations[0])
                                print('进攻')
                                reward += 0.005
                            self.last_attack = iteration

                    except Exception as e:
                        print(e)
                        pass

        elif action == 25:
            if self.supply_army > 0:
                print(f'action={action}')
                for unit in self.units.of_type(
                        {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER,
                         UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
                         UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
                         UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
                         UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
                         UnitTypeId.CHANGELINGZEALOT}):
                    try:
                        '''
                        if iteration - self.last_attack < 200:
                            pass
                        else:
                        '''
                        where2retreat = random.choice(self.units(UnitTypeId.NEXUS))
                        print(self.start_location)
                        unit.move(where2retreat)
                        pass
                        print('撤退')
                    except Exception as e:
                        print(e)
                        pass
        elif action == 26:
            print(f'action={action}')
            if self.structures(UnitTypeId.NEXUS).exists:

                nexus = self.townhalls.random
                if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and self.supply_left >= 2:
                    nexuses = self.structures(UnitTypeId.NEXUS)
                    abilities = await self.get_available_abilities(nexuses)
                    for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                        if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                            loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                            print('星空加速基地')
                            break
        elif action == 27:
            print(f'action={action}')
            if self.structures(UnitTypeId.CYBERNETICSCORE).exists and self.structures(UnitTypeId.NEXUS).exists:

                by = self.structures(UnitTypeId.CYBERNETICSCORE).random
                if not by.is_idle and not by.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                    nexuses = self.structures(UnitTypeId.NEXUS)
                    abilities = await self.get_available_abilities(nexuses)
                    for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                        if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                            loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, by)
                            print('星空加速by')
                            break
        elif action == 28:
            print(f'action={action}')
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists and self.structures(UnitTypeId.NEXUS).exists:

                vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).random
                if not vc.is_idle and not vc.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and self.supply_left >= 4:
                    nexuses = self.structures(UnitTypeId.NEXUS)
                    abilities = await self.get_available_abilities(nexuses)
                    for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                        if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                            loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, vc)
                            print('星空加速vc')
                            break
        elif action == 29:
            print(f'action={action}')
            if self.structures(UnitTypeId.STARGATE).exists and self.structures(UnitTypeId.NEXUS).exists:

                vs = self.structures(UnitTypeId.STARGATE).random
                if not vs.is_idle and not vs.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and self.supply_left >= 4:
                    nexuses = self.structures(UnitTypeId.NEXUS)
                    abilities = await self.get_available_abilities(nexuses)
                    for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                        if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                            loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, vs)
                            print('星空加速vs')
                            break
        elif action == 30:
            print(f'action={action}')
            if self.structures(UnitTypeId.FORGE).exists and self.structures(UnitTypeId.NEXUS).exists:
                bf = self.structures(UnitTypeId.FORGE).random
                if not bf.is_idle and not bf.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and self.supply_left >= 4:
                    nexuses = self.structures(UnitTypeId.NEXUS)
                    abilities = await self.get_available_abilities(nexuses)
                    for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                        if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                            loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, bf)
                            print('星空加速bf')
                            break
        elif action == 31:
            print(f'action={action}')
            if self.structures(UnitTypeId.GATEWAY).exists:

                gate = self.structures(UnitTypeId.GATEWAY).random
                if self.supply_left >= 2 and gate.is_idle and self.can_afford(UnitTypeId.ADEPT) and self.structures(
                        UnitTypeId.CYBERNETICSCORE).exists:
                    gate.train(UnitTypeId.ADEPT)
                    print('训练使徒')
        elif action == 32:
            print(f'action={action}')
            if self.structures(UnitTypeId.GATEWAY).exists:

                gate = self.structures(UnitTypeId.GATEWAY).random
                if self.supply_left >= 2 and gate.is_idle and self.can_afford(UnitTypeId.STALKER) and self.structures(
                        UnitTypeId.CYBERNETICSCORE).exists:
                    gate.train(UnitTypeId.STALKER)
                    print('训练追猎者')
        elif action == 33:
            print(f'action={action}')
            if self.structures(UnitTypeId.GATEWAY).exists:

                gate = self.structures(UnitTypeId.GATEWAY).random
                if self.supply_left >= 2 and gate.is_idle and self.can_afford(UnitTypeId.SENTRY) and self.structures(
                        UnitTypeId.CYBERNETICSCORE).exists:
                    gate.train(UnitTypeId.SENTRY)
                    print('训练哨兵')
        elif action == 34:
            print(f'action={action}')
            if self.structures(UnitTypeId.GATEWAY).exists:

                gate = self.structures(UnitTypeId.GATEWAY).random
                if self.supply_left >= 2 and gate.is_idle and self.can_afford(UnitTypeId.ZEALOT):
                    gate.train(UnitTypeId.ZEALOT)
                    print('训练狂热者')
        elif action == 35:
            print(f'action={action}')
            if self.structures(UnitTypeId.GATEWAY).exists:

                gate = self.structures(UnitTypeId.GATEWAY).random
                if self.supply_left >= 2 and gate.is_idle and self.can_afford(
                        UnitTypeId.HIGHTEMPLAR) and self.structures(UnitTypeId.TEMPLARARCHIVE).exists:
                    gate.train(UnitTypeId.HIGHTEMPLAR)
                    print('训练高阶圣堂武士')
        elif action == 36:
            print(f'action={action}')
            if self.structures(UnitTypeId.GATEWAY).exists:

                gate = self.structures(UnitTypeId.GATEWAY).random
                if self.supply_left >= 2 and gate.is_idle and self.can_afford(
                        UnitTypeId.DARKTEMPLAR) and self.structures(UnitTypeId.DARKSHRINE).exists:
                    gate.train(UnitTypeId.DARKSHRINE)
                    print('训练黑暗圣堂武士')

        elif action == 37:
            print(f'action={action}')
            if self.structures(UnitTypeId.WARPGATE).exists:
                proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

                warpgate = self.structures(UnitTypeId.WARPGATE).random
                abilities = await self.get_available_abilities(warpgate)
                # all the units have the same cooldown anyway so let's just look at ZEALOT
                if AbilityId.WARPGATETRAIN_STALKER in abilities and self.can_afford(
                        UnitTypeId.STALKER) and self.supply_left >= 2:
                    pos = proxy.position.to2.random_on_distance(4)
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                    if placement is None:
                        # return ActionResult.CantFindPlacementLocation
                        logger.info("can't place")
                        return
                    warpgate.warp_in(UnitTypeId.STALKER, placement)
                    print('折跃追猎者')
        elif action == 38:
            print(f'action={action}')
            if self.structures(UnitTypeId.WARPGATE).exists:

                proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

                warpgate = self.structures(UnitTypeId.WARPGATE).random
                abilities = await self.get_available_abilities(warpgate)
                # all the units have the same cooldown anyway so let's just look at ZEALOT
                if AbilityId.WARPGATETRAIN_ZEALOT in abilities and self.can_afford(
                        UnitTypeId.ZEALOT) and self.supply_left >= 2:
                    pos = proxy.position.to2.random_on_distance(4)
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_ZEALOT, pos, placement_step=1)
                    if placement is None:
                        # return ActionResult.CantFindPlacementLocation
                        logger.info("can't place")
                        return
                    warpgate.warp_in(UnitTypeId.ZEALOT, placement)
                    print('折跃狂热者')
        elif action == 39:
            print(f'action={action}')
            if self.structures(UnitTypeId.WARPGATE).exists:

                proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

                warpgate = self.structures(UnitTypeId.WARPGATE).random
                abilities = await self.get_available_abilities(warpgate)
                # all the units have the same cooldown anyway so let's just look at ZEALOT
                if AbilityId.WARPGATETRAIN_HIGHTEMPLAR in abilities and self.can_afford(
                        UnitTypeId.HIGHTEMPLAR) and self.supply_left >= 2:
                    pos = proxy.position.to2.random_on_distance(4)
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_HIGHTEMPLAR, pos, placement_step=1)
                    if placement is None:
                        # return ActionResult.CantFindPlacementLocation
                        logger.info("can't place")
                        return
                    warpgate.warp_in(UnitTypeId.HIGHTEMPLAR, placement)
                    print('折跃高阶圣堂武士')
        elif action == 40:
            print(f'action={action}')
            if self.structures(UnitTypeId.WARPGATE).exists:

                proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

                warpgate = self.structures(UnitTypeId.WARPGATE).random
                abilities = await self.get_available_abilities(warpgate)
                # all the units have the same cooldown anyway so let's just look at ZEALOT
                if AbilityId.WARPGATETRAIN_DARKTEMPLAR in abilities and self.can_afford(
                        UnitTypeId.DARKTEMPLAR) and self.supply_left >= 2:
                    pos = proxy.position.to2.random_on_distance(4)
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_DARKTEMPLAR, pos, placement_step=1)
                    if placement is None:
                        # return ActionResult.CantFindPlacementLocation
                        logger.info("can't place")
                        return
                    warpgate.warp_in(UnitTypeId.DARKTEMPLAR, placement)
                    print('折跃黑暗圣堂武士')
        elif action == 41:
            print(f'action={action}')
            if self.structures(UnitTypeId.WARPGATE).exists:

                proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

                warpgate = self.structures(UnitTypeId.WARPGATE).random
                abilities = await self.get_available_abilities(warpgate)
                # all the units have the same cooldown anyway so let's just look at ZEALOT
                if AbilityId.WARPGATETRAIN_SENTRY in abilities and self.can_afford(
                        UnitTypeId.SENTRY) and self.supply_left >= 2:
                    pos = proxy.position.to2.random_on_distance(4)
                    placement = await self.find_placement(AbilityId.WARPGATETRAIN_SENTRY, pos, placement_step=1)
                    if placement is None:
                        # return ActionResult.CantFindPlacementLocation
                        logger.info("can't place")
                        return
                    warpgate.warp_in(UnitTypeId.SENTRY, placement)
                    print('折跃机械哨兵')
        elif action == 42:
            print(f'action={action}')
            if self.units(UnitTypeId.HIGHTEMPLAR).exists:
                hts = self.units(UnitTypeId.HIGHTEMPLAR)
                for ht in hts:
                    ht(AbilityId.MORPH_ARCHON)
                    print('合成执政官')
        elif action == 43:
            print(f'action={action}')
            if self.structures(UnitTypeId.ROBOTICSFACILITY).exists:
                if self.units(UnitTypeId.OBSERVER).amount + self.already_pending(UnitTypeId.OBSERVER) <= 2:
                    vrs = self.structures(UnitTypeId.ROBOTICSFACILITY)
                    for vr in vrs:
                        if vr.is_idle and self.can_afford(UnitTypeId.OBSERVER) and self.supply_left >= 2:
                            vr.train(UnitTypeId.OBSERVER)
                            print('训练ob')
        elif action == 44:
            print(f'action={action}')
            if self.structures(UnitTypeId.ROBOTICSFACILITY).exists:
                vrs = self.structures(UnitTypeId.ROBOTICSFACILITY)
                for vr in vrs:
                    if vr.is_idle and self.can_afford(UnitTypeId.IMMORTAL) and self.supply_left >= 5:
                        vr.train(UnitTypeId.IMMORTAL)
                        print('训练不朽者')
        elif action == 45:
            print(f'action={action}')
            if self.structures(UnitTypeId.ROBOTICSFACILITY).exists and self.structures(UnitTypeId.ROBOTICSBAY).exists:
                if self.units(UnitTypeId.WARPPRISM).amount + self.already_pending(UnitTypeId.WARPPRISM) <= 1:
                    vrs = self.structures(UnitTypeId.ROBOTICSFACILITY)
                    for vr in vrs:
                        if vr.is_idle and self.can_afford(UnitTypeId.WARPPRISM) and self.supply_left >= 4:
                            vr.train(UnitTypeId.WARPPRISM)
                            print('训练折跃棱镜')
        elif action == 46:
            print(f'action={action}')
            if self.structures(UnitTypeId.ROBOTICSFACILITY).exists and self.structures(UnitTypeId.ROBOTICSBAY).exists:
                vrs = self.structures(UnitTypeId.ROBOTICSFACILITY)
                for vr in vrs:
                    if vr.is_idle and self.can_afford(UnitTypeId.COLOSSUS) and self.supply_left >= 6:
                        vr.train(UnitTypeId.COLOSSUS)
                        print('训练巨像')
        elif action == 47:
            print(f'action={action}')
            if self.structures(UnitTypeId.ROBOTICSFACILITY).exists and self.structures(UnitTypeId.ROBOTICSBAY).exists:
                if self.units(UnitTypeId.DISRUPTOR).amount + self.already_pending(UnitTypeId.DISRUPTOR) <= 1:

                    vrs = self.structures(UnitTypeId.ROBOTICSFACILITY)
                    for vr in vrs:
                        if vr.is_idle and self.can_afford(UnitTypeId.DISRUPTOR) and self.supply_left >= 4:
                            vr.train(UnitTypeId.DISRUPTOR)
                            print('训练自爆球')
        elif action == 48:
            print(f'action={action}')
            if self.structures(UnitTypeId.FORGE).exists and self.structures(UnitTypeId.NEXUS).exists:
                if self.structures(UnitTypeId.NEXUS).amount + self.already_pending(UnitTypeId.NEXUS) <= 2:
                    nexus = self.townhalls.random

                    if self.can_afford(UnitTypeId.PHOTONCANNON) + self.structures(
                            UnitTypeId.PHOTONCANNON).amount + self.already_pending(UnitTypeId.PHOTONCANNON) <= 3:
                        place_position = nexus.position
                        placement_position = await self.find_placement(UnitTypeId.PHOTONCANNON,
                                                                       near=place_position,
                                                                       placement_step=2)
                        if placement_position is not None:
                            await self.build(UnitTypeId.PHOTONCANNON, near=placement_position)
                            print('建设BC')
                else:
                    nexus = self.townhalls.random

                    if self.can_afford(UnitTypeId.PHOTONCANNON) + self.structures(
                            UnitTypeId.PHOTONCANNON).amount + self.already_pending(UnitTypeId.PHOTONCANNON) <= 6:
                        place_position = nexus.position
                        placement_position = await self.find_placement(UnitTypeId.PHOTONCANNON,
                                                                       near=place_position,
                                                                       placement_step=2)
                        if placement_position is not None:
                            await self.build(UnitTypeId.PHOTONCANNON, near=placement_position)
                            print('建设BC')


        elif action == 49:
            print(f'action={action}')
            if self.structures(UnitTypeId.CYBERNETICSCORE).exists and self.structures(UnitTypeId.NEXUS).exists:
                if self.structures(UnitTypeId.NEXUS).amount + self.already_pending(UnitTypeId.NEXUS) <= 2:
                    nexus = self.townhalls.random

                    if self.can_afford(UnitTypeId.SHIELDBATTERY) and self.structures(
                            UnitTypeId.SHIELDBATTERY).amount + self.already_pending(UnitTypeId.SHIELDBATTERY) <= 1:
                        place_position = nexus.position
                        placement_position = await self.find_placement(UnitTypeId.SHIELDBATTERY,
                                                                       near=place_position,
                                                                       placement_step=2)
                        if placement_position is not None:
                            await self.build(UnitTypeId.SHIELDBATTERY, near=placement_position)
                            print('建设BB')
                else:
                    nexus = self.townhalls.random

                    if self.can_afford(UnitTypeId.SHIELDBATTERY) and self.structures(
                            UnitTypeId.SHIELDBATTERY).amount + self.already_pending(UnitTypeId.SHIELDBATTERY) <= 3:
                        place_position = nexus.position
                        placement_position = await self.find_placement(UnitTypeId.SHIELDBATTERY,
                                                                       near=place_position,
                                                                       placement_step=2)
                        if placement_position is not None:
                            await self.build(UnitTypeId.SHIELDBATTERY, near=placement_position)
                            print('建设BB')



        # do nothing
        elif action == 50:
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
                            {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.VOIDRAY, UnitTypeId.STALKER,
                             UnitTypeId.ADEPT,
                             UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR, UnitTypeId.OBSERVER,
                             UnitTypeId.CHANGELINGZEALOT}):
                        closed_enemy = self.enemy_units.sorted(lambda x: x.distance_to(unit))
                        unit.attack(closed_enemy[0])
                else:
                    self.rally_defend = False

            if self.rally_defend == True:
                map_center = self.game_info.map_center
                rally_point = self.townhalls.random.position.towards(map_center, distance=5)
                for unit in self.units.of_type(
                        {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.VOIDRAY, UnitTypeId.STALKER, UnitTypeId.ADEPT,
                         UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR, UnitTypeId.OBSERVER,
                         UnitTypeId.CHANGELINGZEALOT}):
                    if unit.distance_to(self.start_location) > 100 and unit not in self.unit_tags_received_action:
                        unit.move(rally_point)


def worker(transaction, lock):
    laddermap_2023 = ['Altitude LE', 'Ancient Cistern LE', 'Babylon LE', 'Dragon Scales LE', 'Gresvan LE',
                      'Neohumanity LE', 'Royal Blood LE']
    res = run_game(maps.get(laddermap_2023[0]),
                   [Bot(Race.Protoss, WorkerRunshBot(transaction, lock)), Computer(Race.Terran, Difficulty.Easy)],
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
