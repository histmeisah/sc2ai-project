import random
from typing import Set

from loguru import logger
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.units import Units

ladder_map_pool_2022_07 = ["Data-C", "Moondance", "Stargazers", "Waterfall", "Tropical Sacrifice", "Inside and Out",
                           "Cosmic Sapphire"]


# pylint: disable=W0231
class WarpGateBot(BotAI):

    def __init__(self):
        # Initialize inherited class
        self.proxy_built = False
        self.main_base = None
        self.second_base = None
        self.third_base = None
        self.forth_base = None
        self.produce_zealot = False
        self.produce_stalker = False
        self.produce_high_templar = False
        self.army_units: Units = []
        self.worker_tag_list = []
        self.upgrade_tag_list = []
        self.flag = True
        self.add_on = False
        self.attacking = False
        self.auto_mode = False
        self.vespenetrigger = False
        self.rally_defend = True
        self.morph_archon = False
        self.zealot_train = False
        self.stalker_train = False
        self.ht_train = False
        self.dt_train = False
        self.sentry_train = False
        self.expansion_flag= False
        self.proxy_flag = False
        self.p=None
    # pylint: disable=R0912
    async def on_step(self, iteration):
        await self.procedure()
        await self.distribute_workers()
        await self.defend()
        await self.attack()
        if self.auto_mode == True:
            await self.building_supply()
            await self.expansion()
            await self.produce_worker()
            await self.build_vespene()
        if self.workers.amount >= 66:
            await self.CHRONOBOOSTENERGYCOST_upgrade()
        if self.expansion_flag==True:
            await self.expansion()
        if self.zealot_train==True:
            await self.train_zealot()
        if self.stalker_train==True:
            await self.train_stalker()
        if self.ht_train==True:
            await self.train_ht()
            await self.train_archon()
        if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready and self.structures(UnitTypeId.FORGE).ready and self.workers.amount>=66:
            await self.CHRONOBOOSTENERGYCOST_upgrade()
        await self.stalker_blink()
        await self.research_blink()
    async def procedure(self):
        if self.time_formatted == '00:00':
            if self.start_location == Point2((160.5, 46.5)):
                self.Location = -1  # detect location
            else:
                self.Location = 1
            await self.chat_send("(glhf)")
            self.main_base = self.townhalls.first
        elif '00:00' <= self.time_formatted <= '01:00':
            nexus = self.townhalls.first
            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):
                nexus.train(UnitTypeId.PROBE)
            if '00:17' <= self.time_formatted <= '00:25':
                if self.supply_used <= 15 and self.already_pending(UnitTypeId.PYLON) + self.structures(
                        UnitTypeId.PYLON).amount == 0:
                    if self.can_afford(UnitTypeId.PYLON):
                        await self.build(UnitTypeId.PYLON, near=nexus)
            elif '00:20' <= self.time_formatted <= '01:00':
                if '00:38' <= self.time_formatted <= '00:42':
                    pylon = self.structures(UnitTypeId.PYLON).random
                    if self.supply_used == 16 and self.already_pending(UnitTypeId.GATEWAY) + self.structures(
                            UnitTypeId.GATEWAY).amount == 0:
                        if self.can_afford(UnitTypeId.GATEWAY):
                            await self.build(UnitTypeId.GATEWAY, near=pylon, placement_step=2)
                if '00:40' <= self.time_formatted <= '01:00' and self.already_pending(UnitTypeId.GATEWAY) == 1:
                    if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        nexuses = self.structures(UnitTypeId.NEXUS)
                        abilities = await self.get_available_abilities(nexuses)
                        for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                                loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                                break
                    if '00:48' <= self.time_formatted <= '00:51' and self.structures(
                            UnitTypeId.ASSIMILATOR).amount + self.already_pending(UnitTypeId.ASSIMILATOR) < 1:
                        if 16 <= self.workers.amount <= 20 and self.can_afford(
                                UnitTypeId.ASSIMILATOR) and self.already_pending(UnitTypeId.GATEWAY) == 1:
                            vespenes = self.vespene_geyser.closer_than(15, self.main_base).random
                            if self.can_afford(UnitTypeId.ASSIMILATOR):
                                await self.build(UnitTypeId.ASSIMILATOR, vespenes)
        elif '01:00' <= self.time_formatted <= '02:00':
            nexus = self.townhalls.first
            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE) and self.already_pending(UnitTypeId.PROBE) == 0:
                nexus.train(UnitTypeId.PROBE)
            if '01:25' <= self.time_formatted <= '01:40' and self.structures(UnitTypeId.GATEWAY).ready:
                if self.can_afford(UnitTypeId.NEXUS) and 19 <= self.workers.amount <= 22:
                    if self.townhalls.amount + self.already_pending(UnitTypeId.NEXUS) < 2:
                        await self.expand_now()
            if '01:39' <= self.time_formatted <= '01:44' and self.structures(
                    UnitTypeId.CYBERNETICSCORE).amount + self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0:
                pylon = self.structures(UnitTypeId.PYLON).random
                if self.structures(UnitTypeId.GATEWAY).ready and self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.already_pending(UnitTypeId.NEXUS):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
            elif '01:44' <= self.time_formatted <= '01:59':

                if 20 <= self.supply_used <= 22 and self.structures(
                        UnitTypeId.ASSIMILATOR).ready and self.already_pending(
                    UnitTypeId.CYBERNETICSCORE) == 1 and self.already_pending(UnitTypeId.NEXUS) == 1:
                    if self.can_afford(UnitTypeId.ASSIMILATOR):
                        for nexus in self.townhalls.ready:
                            vgs = self.vespene_geyser.closer_than(15, nexus)
                            for vg in vgs:
                                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                                    break
                                worker = self.select_build_worker(vg.position)
                                if worker is None:
                                    break
                                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vg).exists:
                                    worker.build(UnitTypeId.ASSIMILATOR, vg)
                if self.already_pending(UnitTypeId.PYLON) + self.structures(UnitTypeId.PYLON).amount == 1:
                    if self.supply_left <= 2:
                        if self.can_afford(UnitTypeId.PYLON):
                            await self.build(UnitTypeId.PYLON, near=nexus)
        elif '02:00' <= self.time_formatted <= '03:00':
            bases = self.townhalls
            for base in bases:
                if base.is_idle and self.can_afford(UnitTypeId.PROBE):
                    base.train(UnitTypeId.PROBE)
            if '02:10' <= self.time_formatted <= '02:40':
                if self.structures(UnitTypeId.PYLON).amount == 2 and self.supply_left >= 3:
                    nexus = self.main_base
                    if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        nexuses = self.structures(UnitTypeId.NEXUS)
                        abilities = await self.get_available_abilities(nexuses)
                        for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                                loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                                break
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.can_afford(UnitTypeId.TWILIGHTCOUNCIL) and not self.already_pending(UnitTypeId.ADEPT):
                    pylon = self.structures(UnitTypeId.PYLON).random
                    if self.structures(UnitTypeId.STARGATE).amount + self.already_pending(
                            UnitTypeId.STARGATE) == 0:
                        await self.build(UnitTypeId.STARGATE, near=pylon)
                if (
                        self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.can_afford(
                    AbilityId.RESEARCH_WARPGATE)
                        and self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0
                ):
                    ccore = self.structures(UnitTypeId.CYBERNETICSCORE).ready.first
                    ccore.research(UpgradeId.WARPGATERESEARCH)
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UnitTypeId.STARGATE) != 0 and self.can_afford(UnitTypeId.ADEPT):
                    if self.structures(UnitTypeId.GATEWAY).idle:
                        gate = self.structures(UnitTypeId.GATEWAY).random
                        if self.supply_left >= 3 and self.can_afford(UnitTypeId.ADEPT) and self.already_pending(
                                UnitTypeId.ADEPT) + self.units(UnitTypeId.ADEPT).amount <= 1:
                            gate.train(UnitTypeId.ADEPT)
                if self.already_pending(UnitTypeId.ADEPT) and not self.structures(UnitTypeId.GATEWAY).idle:
                    gate = self.structures(UnitTypeId.GATEWAY).random
                    if not gate.is_idle and not gate.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        nexuses = self.structures(UnitTypeId.NEXUS)
                        abilities = await self.get_available_abilities(nexuses)
                        for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                                loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, gate)
                                break
                if self.already_pending(UnitTypeId.ADEPT) and self.already_pending(
                        UnitTypeId.STARGATE) and self.already_pending(UnitTypeId.STARGATE) and not self.already_pending(
                    UpgradeId.WARPGATERESEARCH):
                    if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.can_afford(
                            UpgradeId.WARPGATERESEARCH):
                        by = self.structures(UnitTypeId.CYBERNETICSCORE).random
                        if by.is_idle:
                            by.research(UpgradeId.WARPGATERESEARCH)
            elif '02:40' <= self.time_formatted <= '03:00':
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UnitTypeId.STARGATE) != 0 and self.can_afford(UnitTypeId.STALKER):
                    if self.structures(UnitTypeId.GATEWAY).idle and self.units(
                            UnitTypeId.STALKER).amount + self.already_pending(UnitTypeId.STALKER) <= 1:
                        gate = self.structures(UnitTypeId.GATEWAY).random
                        if self.supply_left >= 3 and self.can_afford(UnitTypeId.STALKER):
                            gate.train(UnitTypeId.STALKER)

                bases = self.townhalls
                for base in bases:
                    if base != self.main_base:
                        self.second_base = base
                if self.second_base != None:
                    if self.structures(UnitTypeId.NEXUS).amount == 2:
                        if not self.second_base.is_idle and not self.second_base.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexuses = self.structures(UnitTypeId.NEXUS)
                            abilities = await self.get_available_abilities(nexuses)
                            for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                                    loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, self.second_base)
                                    break
        elif '03:00' <= self.time_formatted <= '04:00':

            bases = self.townhalls
            for base in bases:
                if base.is_idle and self.can_afford(UnitTypeId.PROBE):
                    base.train(UnitTypeId.PROBE)

            if '03:00' <= self.time_formatted <= '03:15':
                vs = self.structures(UnitTypeId.STARGATE).random
                if self.structures(UnitTypeId.STARGATE).ready and self.structures(UnitTypeId.NEXUS).amount == 2:
                    if self.structures(UnitTypeId.STARGATE).idle and self.can_afford(
                            UnitTypeId.VOIDRAY) and self.supply_left >= 4:
                        if vs.is_idle and self.can_afford(UnitTypeId.VOIDRAY):
                            vs.train(UnitTypeId.VOIDRAY)
                if not self.structures(UnitTypeId.STARGATE).idle and self.already_pending(UnitTypeId.VOIDRAY):
                    if self.structures(UnitTypeId.NEXUS).amount == 2:
                        if not vs.is_idle and not vs.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                            nexuses = self.structures(UnitTypeId.NEXUS)
                            abilities = await self.get_available_abilities(nexuses)
                            for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                                    loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, vs)
                                    break
                if self.already_pending(UnitTypeId.VOIDRAY) and self.supply_used >= 39 and self.already_pending(
                        UnitTypeId.PYLON) == 0:
                    if self.can_afford(UnitTypeId.PYLON):
                        place_postion = self.second_base.position + Point2((0, self.Location * 8))
                        await self.build(UnitTypeId.PYLON, near=place_postion, placement_step=3)

            elif '03:15' <= self.time_formatted <= '03:50':
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(
                        UnitTypeId.STARGATE) != 0 and self.can_afford(UnitTypeId.ADEPT):
                    if self.structures(UnitTypeId.GATEWAY).idle:
                        gate = self.structures(UnitTypeId.GATEWAY).random
                        if self.supply_left >= 3 and self.can_afford(UnitTypeId.STALKER) and self.already_pending(
                                UnitTypeId.STALKER) + self.units(UnitTypeId.STALKER).amount <= 2:
                            gate.train(UnitTypeId.STALKER)
                if self.structures(UnitTypeId.GATEWAY).amount + self.already_pending(UnitTypeId.GATEWAY) == 1:
                    await self.build(UnitTypeId.GATEWAY, near=self.main_base)

                if self.supply_left<=4 and self.can_afford(UnitTypeId.PYLON) and self.already_pending(UnitTypeId.PYLON)==0:
                    place_postion = self.second_base.position
                    await self.build(UnitTypeId.PYLON, near=place_postion)
                nexuses = self.structures(UnitTypeId.NEXUS)
                nexus = nexuses.random
                if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                    abilities = await self.get_available_abilities(nexuses)
                    for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                        if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                            loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                            break
                if self.supply_left <= 4 and self.can_afford(UnitTypeId.PYLON) and self.already_pending(
                        UnitTypeId.NEXUS):
                    if self.can_afford(UnitTypeId.PYLON):
                        await self.build(UnitTypeId.PYLON, near=self.second_base)
            elif '03:50' <= self.time_formatted <= '04:00':
                if self.already_pending(UnitTypeId.NEXUS) and self.supply_left >= 4:
                    if self.structures(UnitTypeId.STARGATE).idle and self.can_afford(UnitTypeId.VOIDRAY):
                        vs = self.structures(UnitTypeId.STARGATE).random
                        if vs.is_idle and self.can_afford(UnitTypeId.VOIDRAY):
                            vs.train(UnitTypeId.VOIDRAY)
        elif '04:00' <= self.time_formatted <= '05:00':
            bases = self.townhalls
            for base in bases:
                if base.is_idle and self.can_afford(UnitTypeId.PROBE) and self.supply_left >= 3:
                    base.train(UnitTypeId.PROBE)
            if '04:00' <= self.time_formatted <= '04:20':
                if self.structures(
                        UnitTypeId.PYLON).amount <= 4 and self.supply_used >= 48 and self.townhalls.amount + self.already_pending(
                        UnitTypeId.NEXUS) < 3:
                    if self.can_afford(UnitTypeId.NEXUS) and self.already_pending(
                            UnitTypeId.NEXUS) == 0:
                        if self.can_afford(UnitTypeId.NEXUS):
                            await self.expand_now()
                if self.supply_left <= 4 and self.can_afford(UnitTypeId.PYLON) and self.already_pending(
                        UnitTypeId.PYLON) < 2:
                    place_postion = self.second_base.position + Point2((0, self.Location * 8))
                    await self.build(UnitTypeId.PYLON, near=place_postion, placement_step=2)
                if self.already_pending(UnitTypeId.NEXUS) and self.can_afford(UnitTypeId.ASSIMILATOR):
                    nexus = self.second_base
                    vgs = self.vespene_geyser.closer_than(15, nexus)
                    for vg in vgs:
                        if not self.can_afford(UnitTypeId.ASSIMILATOR):
                            break
                        worker = self.select_build_worker(vg.position)
                        if worker is None:
                            break
                        if not self.gas_buildings or not self.gas_buildings.closer_than(1, vg):
                            worker.build_gas(vg)
                            worker.stop(queue=True)
                if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.can_afford(
                        UnitTypeId.SHIELDBATTERY) and self.structures(
                        UnitTypeId.SHIELDBATTERY).amount + self.already_pending(UnitTypeId.SHIELDBATTERY) == 0:
                        await self.build(UnitTypeId.SHIELDBATTERY, near=self.main_base)
            elif '04:20' <= self.time_formatted <= '05:00':
                if self.structures(UnitTypeId.TWILIGHTCOUNCIL).amount + self.already_pending(
                        UnitTypeId.TWILIGHTCOUNCIL) == 0:
                    if self.can_afford(UnitTypeId.TWILIGHTCOUNCIL):
                        pylon = self.structures(UnitTypeId.PYLON).random
                        await self.build(UnitTypeId.TWILIGHTCOUNCIL, near=pylon)
                if self.structures(UnitTypeId.FORGE).amount + self.already_pending(UnitTypeId.FORGE) == 0:
                    if self.can_afford(UnitTypeId.FORGE):
                        pylon = self.structures(UnitTypeId.PYLON).random
                        await self.build(UnitTypeId.FORGE, near=pylon)
                if self.supply_left <= 4 and self.can_afford(UnitTypeId.PYLON) and self.already_pending(
                        UnitTypeId.PYLON) < 2:
                    place_postion = self.second_base.position + Point2((0, self.Location * 8))
                    await self.build(UnitTypeId.PYLON, near=place_postion, placement_step=2)
                if self.can_afford(UnitTypeId.STALKER) and self.supply_left >= 4:
                    if self.units(UnitTypeId.STALKER).amount < 4:
                        proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])
                        await self.warp_stalker()
                if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left >= 4 and self.units(
                        UnitTypeId.VOIDRAY).amount < 3:
                    vs = self.structures(UnitTypeId.STARGATE).random
                    vs.train(UnitTypeId.VOIDRAY)

        elif '05:00' <= self.time_formatted <= '07:00':
            if self.townhalls.amount>=3 and self.third_base==None and self.forth_base==None:
                nexuses = self.townhalls
                for nexus in nexuses:
                    if nexus!=self.main_base and nexus!= self.second_base:
                        self.third_base=nexus
            self.expansion_flag=True
            vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).first
            bf = self.structures(UnitTypeId.FORGE).random
            if self.townhalls.amount >= 3:
                self.auto_mode = True
                self.zealot_train = True
            if self.units(UnitTypeId.STALKER).amount+self.already_pending(UnitTypeId.STALKER)<=8:
                await self.warp_stalker()
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready:
                if self.structures(UnitTypeId.TEMPLARARCHIVE).amount + self.already_pending(
                        UnitTypeId.TEMPLARARCHIVE) == 0:
                    await self.build(UnitTypeId.TEMPLARARCHIVE, near=self.second_base)
                if self.can_afford(UpgradeId.CHARGE) and vc.is_idle:
                    vc.research(UpgradeId.CHARGE)
            if self.structures(UnitTypeId.FORGE).ready:
                if self.can_afford(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1):
                    bf.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1)
            if self.structures(UnitTypeId.TEMPLARARCHIVE).ready:
                self.stalker_train = True
        elif '07:00'<=self.time_formatted:
            if self.proxy_flag==False and self.p==None:
                p = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)
                self.p=p
                await self.build(UnitTypeId.PYLON, near=p)
                self.proxy_flag=True
            if self.townhalls.amount>=4 and self.forth_base == None:
                nexuses = self.townhalls
                for nexus in nexuses:
                    if nexus!=self.main_base and nexus!= self.second_base and nexus!=self.third_base:
                        self.forth_base=nexus

            elif '07:00' <= self.time_formatted  and self.can_afford(UnitTypeId.ZEALOT):
                if self.units(UnitTypeId.ZEALOT).amount + self.already_pending(
                        UnitTypeId.ZEALOT) < 25 and self.supply_left >= 2:
                    await self.warp_zealot()


    async def train_zealot(self):
        if self.structures(UnitTypeId.PYLON).exists and self.can_afford(UnitTypeId.ZEALOT):
            if '05:00' <= self.time_formatted <= '07:00' and self.workers.amount >= 44:
                if self.units(UnitTypeId.ZEALOT).amount + self.already_pending(
                        UnitTypeId.ZEALOT) < 10 and self.supply_left >= 2:
                    await self.warp_zealot()
            elif '07:00' <= self.time_formatted <= '09:00' and self.can_afford(UnitTypeId.ZEALOT):
                if self.units(UnitTypeId.ZEALOT).amount + self.already_pending(
                        UnitTypeId.ZEALOT) < 35 and self.supply_left >= 2:
                    await self.warp_zealot()
            elif '09:00' <= self.time_formatted and self.can_afford(UnitTypeId.ZEALOT):
                if self.minerals>=2000:
                    if self.units(UnitTypeId.ZEALOT).amount + self.already_pending(
                            UnitTypeId.ZEALOT) < 50 and self.supply_left >= 2:
                        await self.warp_zealot()

                if self.units(UnitTypeId.ZEALOT).amount + self.already_pending(
                        UnitTypeId.ZEALOT) < 30 and self.supply_left >= 2:
                    await self.warp_zealot()
    async def train_stalker(self):
        if self.structures(UnitTypeId.PYLON).exists and self.can_afford(UnitTypeId.STALKER):
            if '05:00' <= self.time_formatted <= '07:00' and self.workers.amount >= 44:
                if self.units(UnitTypeId.STALKER).amount + self.already_pending(
                        UnitTypeId.STALKER) < 8 and self.supply_left >= 2:
                    await self.warp_stalker()
            elif '07:00' <= self.time_formatted <= '09:00' and self.can_afford(UnitTypeId.STALKER):
                if self.units(UnitTypeId.STALKER).amount + self.already_pending(
                        UnitTypeId.STALKER) < 15 and self.supply_left >= 2:
                    await self.warp_stalker()
            elif '09:00' <= self.time_formatted and self.can_afford(UnitTypeId.STALKER):
                if self.units(UnitTypeId.STALKER).amount + self.already_pending(
                        UnitTypeId.STALKER) < 100 and self.supply_left >= 2:
                    await self.warp_stalker()

    async def train_ht(self):
        if self.structures(UnitTypeId.PYLON).exists and self.can_afford(UnitTypeId.HIGHTEMPLAR):
            if '05:00' <= self.time_formatted <= '07:00' and self.workers.amount >= 60:
                if self.units(UnitTypeId.HIGHTEMPLAR).amount + self.already_pending(UnitTypeId.HIGHTEMPLAR) + \
                        self.units(UnitTypeId.ARCHON).amount * 2 + self.already_pending(
                    UnitTypeId.ARCHON) * 2 < 4 and self.supply_left >= 2:
                    await self.warp_ht()
            elif '07:00' <= self.time_formatted <= '09:00' and self.can_afford(UnitTypeId.HIGHTEMPLAR):
                if self.units(UnitTypeId.HIGHTEMPLAR).amount + self.already_pending(UnitTypeId.HIGHTEMPLAR) + \
                        self.units(UnitTypeId.ARCHON).amount * 2 + self.already_pending(
                    UnitTypeId.ARCHON) * 2 < 8 and self.supply_left >= 2:
                    await self.warp_ht()

            elif '09:00' <= self.time_formatted and self.can_afford(UnitTypeId.HIGHTEMPLAR):
                if self.units(UnitTypeId.HIGHTEMPLAR).amount + self.already_pending(UnitTypeId.HIGHTEMPLAR) + \
                        self.units(UnitTypeId.ARCHON).amount * 2 + self.already_pending(
                    UnitTypeId.ARCHON) * 2 < 14 and self.supply_left >= 2:
                    await self.warp_ht()

    async def train_dt(self):
        if self.structures(UnitTypeId.PYLON).exists and self.can_afford(UnitTypeId.DARKTEMPLAR):
            if '05:00' <= self.time_formatted <= '07:00' and self.workers.amount >= 30:
                if self.units(UnitTypeId.DARKTEMPLAR).amount + self.already_pending(
                        UnitTypeId.DARKTEMPLAR) < 4 and self.supply_left >= 2:
                    await self.warp_dt()
            elif '07:00' <= self.time_formatted <= '09:00' and self.can_afford(UnitTypeId.STALKER):
                if self.units(UnitTypeId.DARKTEMPLAR).amount + self.already_pending(
                        UnitTypeId.DARKTEMPLAR) < 6 and self.supply_left >= 2:
                    await self.warp_dt()
            elif '09:00' <= self.time_formatted and self.can_afford(UnitTypeId.DARKTEMPLAR):
                if self.units(UnitTypeId.DARKTEMPLAR).amount + self.already_pending(
                        UnitTypeId.DARKTEMPLAR) < 8 and self.supply_left >= 2:
                    await self.warp_dt()

    async def train_sentry(self):
        if self.structures(UnitTypeId.PYLON).exists and self.can_afford(UnitTypeId.STALKER):
            if '05:00' <= self.time_formatted <= '07:00' and self.workers.amount >= 30:
                if self.units(UnitTypeId.SENTRY).amount + self.already_pending(
                        UnitTypeId.SENTRY) < 2 and self.supply_left >= 2:
                    await self.warp_sentry()
            elif '07:00' <= self.time_formatted <= '09:00' and self.can_afford(UnitTypeId.SENTRY):
                if self.units(UnitTypeId.SENTRY).amount + self.already_pending(
                        UnitTypeId.SENTRY) < 7 and self.supply_left >= 2:
                    await self.warp_sentry()
            elif '09:00' <= self.time_formatted and self.can_afford(UnitTypeId.SENTRY):
                if self.units(UnitTypeId.SENTRY).amount + self.already_pending(
                        UnitTypeId.SENTRY) < 2 and self.supply_left >= 2:
                    await self.warp_sentry()

    async def train_archon(self):
        if self.units(UnitTypeId.HIGHTEMPLAR).exists:
            hts = self.units(UnitTypeId.HIGHTEMPLAR)
            for ht in hts:
                ht(AbilityId.MORPH_ARCHON)

    async def warp_stalker(self):
        proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

        warpgate = self.structures(UnitTypeId.WARPGATE).random
        abilities = await self.get_available_abilities(warpgate)
        # all the units have the same cooldown anyway so let's just look at ZEALOT
        if AbilityId.WARPGATETRAIN_STALKER in abilities and self.can_afford(UnitTypeId.STALKER):
            pos = proxy.position.to2.random_on_distance(4)
            placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
            if placement is None:
                # return ActionResult.CantFindPlacementLocation
                logger.info("can't place")
                return
            warpgate.warp_in(UnitTypeId.STALKER, placement)

    async def warp_zealot(self):
        proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

        warpgate = self.structures(UnitTypeId.WARPGATE).random
        abilities = await self.get_available_abilities(warpgate)
        # all the units have the same cooldown anyway so let's just look at ZEALOT
        if AbilityId.WARPGATETRAIN_ZEALOT in abilities and self.can_afford(UnitTypeId.ZEALOT):
            pos = proxy.position.to2.random_on_distance(4)
            placement = await self.find_placement(AbilityId.WARPGATETRAIN_ZEALOT, pos, placement_step=1)
            if placement is None:
                # return ActionResult.CantFindPlacementLocation
                logger.info("can't place")
                return
            warpgate.warp_in(UnitTypeId.ZEALOT, placement)

    async def warp_ht(self):
        proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

        warpgate = self.structures(UnitTypeId.WARPGATE).random
        abilities = await self.get_available_abilities(warpgate)
        # all the units have the same cooldown anyway so let's just look at ZEALOT
        if AbilityId.WARPGATETRAIN_HIGHTEMPLAR in abilities and self.can_afford(UnitTypeId.HIGHTEMPLAR):
            pos = proxy.position.to2.random_on_distance(4)
            placement = await self.find_placement(AbilityId.WARPGATETRAIN_HIGHTEMPLAR, pos, placement_step=1)
            if placement is None:
                # return ActionResult.CantFindPlacementLocation
                logger.info("can't place")
                return
            warpgate.warp_in(UnitTypeId.HIGHTEMPLAR, placement)

    async def warp_dt(self):
        proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

        warpgate = self.structures(UnitTypeId.WARPGATE).random
        abilities = await self.get_available_abilities(warpgate)
        # all the units have the same cooldown anyway so let's just look at ZEALOT
        if AbilityId.WARPGATETRAIN_DARKTEMPLAR in abilities and self.can_afford(UnitTypeId.DARKTEMPLAR):
            pos = proxy.position.to2.random_on_distance(4)
            placement = await self.find_placement(AbilityId.WARPGATETRAIN_DARKTEMPLAR, pos, placement_step=1)
            if placement is None:
                # return ActionResult.CantFindPlacementLocation
                logger.info("can't place")
                return
            warpgate.warp_in(UnitTypeId.DARKTEMPLAR, placement)

    async def warp_sentry(self):
        proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0])

        warpgate = self.structures(UnitTypeId.WARPGATE).random
        abilities = await self.get_available_abilities(warpgate)
        # all the units have the same cooldown anyway so let's just look at ZEALOT
        if AbilityId.WARPGATETRAIN_SENTRY in abilities and self.can_afford(UnitTypeId.SENTRY):
            pos = proxy.position.to2.random_on_distance(4)
            placement = await self.find_placement(AbilityId.WARPGATETRAIN_SENTRY, pos, placement_step=1)
            if placement is None:
                # return ActionResult.CantFindPlacementLocation
                logger.info("can't place")
                return
            warpgate.warp_in(UnitTypeId.SENTRY, placement)

    async def produce_worker(self):
        if self.townhalls and self.supply_workers <= 67 and self.supply_left >= 3:
            base = self.townhalls.random
            if base.is_idle and self.can_afford(UnitTypeId.PROBE):
                base.train(UnitTypeId.PROBE)
        nexus = self.townhalls.random
        if not nexus.is_idle and not nexus.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
            nexuses = self.structures(UnitTypeId.NEXUS)
            abilities = await self.get_available_abilities(nexuses)
            for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                    loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, nexus)
                    break

    async def CHRONOBOOSTENERGYCOST_upgrade(self):
        if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready:
            vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).random
            if not vc.is_idle and not vc.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                nexuses = self.structures(UnitTypeId.NEXUS)
                abilities = await self.get_available_abilities(nexuses)
                for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                    if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                        loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, vc)

        if self.structures(UnitTypeId.FORGE).ready:
            bf = self.structures(UnitTypeId.FORGE).random
            if not bf.is_idle and not bf.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                nexuses = self.structures(UnitTypeId.NEXUS)
                abilities = await self.get_available_abilities(nexuses)
                for loop_nexus, abilities_nexus in zip(nexuses, abilities):
                    if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                        loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, bf)

    async def build_vespene(self):
        for nexus in self.structures(UnitTypeId.NEXUS).ready:
            # gases = self.state.vespene_geyser.closer_than(9.0, cc)
            gases = self.vespene_geyser.closer_than(10, nexus)
            for gas in gases:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    
                    break
                worker = self.select_build_worker(gas.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.REFINERY).closer_than(1.0, gas).exists:
                    worker.build(UnitTypeId.ASSIMILATOR, gas)

    async def building_supply(self):
        if '05:00' <= self.time_formatted <= '06:00':
            if self.supply_left <= 3 and  self.already_pending(UnitTypeId.PYLON)<=2 and not self.supply_cap == 200:
                if self.can_afford(UnitTypeId.PYLON):
                    place_postion = self.main_base.position + Point2((0, self.Location * 8))
                    await self.build(UnitTypeId.PYLON, near=place_postion, placement_step=2)
        if '06:00' <= self.time_formatted <= '07:00':
            if self.supply_left <= 5 and self.already_pending(UnitTypeId.PYLON) <= 4 and not self.supply_cap == 200:
                if self.can_afford(UnitTypeId.PYLON):
                    place_postion = self.second_base.position + Point2((0, self.Location * 8))
                    await self.build(UnitTypeId.PYLON, near=place_postion, placement_step=2)
        if '06:00' <= self.time_formatted <= '08:00':
            if self.supply_left <= 5 and self.already_pending(
                    UnitTypeId.PYLON) <= 3 and not self.supply_cap == 200:
                if self.can_afford(UnitTypeId.PYLON):
                    place_postion = self.third_base.position
                    await self.build(UnitTypeId.PYLON, near=place_postion, placement_step=2)
        if '08:00' <= self.time_formatted <= '10:00':
            if self.supply_left <= 7 and self.already_pending(
                    UnitTypeId.PYLON) <= 4 and not self.supply_cap  == 200:
                if self.can_afford(UnitTypeId.PYLON):
                    place_postion = self.forth_base.position
                    await self.build(UnitTypeId.PYLON, near=place_postion, placement_step=2)
        if '10:00' <= self.time_formatted:
            if self.supply_left <= 7 and self.already_pending(
                    UnitTypeId.PYLON) <= 4 and not self.supply_cap== 200:
                if self.can_afford(UnitTypeId.PYLON):
                    base = self.townhalls.random
                    place_postion = base.position + Point2((0, self.Location * 8))
                    await self.build(UnitTypeId.PYLON, near=place_postion, placement_step=2)

    async def expansion(self):
        if self.time_formatted <= '09:00':
            if self.minerals > 1000:
                if not self.already_pending(UnitTypeId.NEXUS) and self.can_afford(
                        UnitTypeId.NEXUS) and self.townhalls.amount + self.already_pending(UnitTypeId.NEXUS) < 5:
                    location = await self.get_next_expansion()
                    if location:
                        worker = self.select_build_worker(location)
                        if worker and self.can_afford(UnitTypeId.NEXUS):
                            worker.build(UnitTypeId.NEXUS, location)
        elif self.time_formatted >= '09:00':
            if self.minerals > 1000:
                if not self.already_pending(UnitTypeId.NEXUS) and self.can_afford(
                        UnitTypeId.COMMANDCENTER) and self.townhalls.amount + self.already_pending(
                        UnitTypeId.NEXUS) < 7:
                    location = await self.get_next_expansion()
                    if location:
                        worker = self.select_build_worker(location)
                        if worker and self.can_afford(UnitTypeId.NEXUS):
                            worker.build(UnitTypeId.NEXUS, location)
        if self.minerals > 500 and self.vespene > 300 and self.time_formatted <= '08:00':
            if self.structures(UnitTypeId.GATEWAY).amount + self.structures(
                    UnitTypeId.WARPGATE).amount + self.already_pending(UnitTypeId.GATEWAY) < 10 and self.already_pending(
                    UnitTypeId.GATEWAY) <= 4 and self.can_afford(UnitTypeId.GATEWAY):
                if self.can_afford(UnitTypeId.GATEWAY):
                    pylon = self.structures(UnitTypeId.PYLON).random.position
                    placement_position = await self.find_placement(UnitTypeId.GATEWAY, near=pylon,
                                                                   placement_step=2)
                    worker_candidates = self.workers.filter(lambda worker: (
                                                                                   worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)
                    if placement_position:
                        build_worker = worker_candidates.closest_to(placement_position)
                        build_worker.build(UnitTypeId.GATEWAY, placement_position)
        if self.minerals > 700 and self.vespene > 400 and '08:00' <= self.time_formatted <= '12:00':
            if self.structures(UnitTypeId.GATEWAY).amount + self.structures(
                    UnitTypeId.WARPGATE).amount + self.already_pending(
                    UnitTypeId.GATEWAY) < 14 and self.already_pending(
                    UnitTypeId.GATEWAY) <= 4 and self.can_afford(UnitTypeId.GATEWAY):
                if self.can_afford(UnitTypeId.GATEWAY) and self.third_base:
                    building_place = self.structures(UnitTypeId.PYLON).closest_to(self.third_base).position
                    placement_position = await self.find_placement(UnitTypeId.GATEWAY, near=building_place,
                                                                   placement_step=2)
                    worker_candidates = self.workers.filter(lambda worker: (
                                                                                   worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)
                    if placement_position:
                        build_worker = worker_candidates.closest_to(placement_position)
                        build_worker.build(UnitTypeId.GATEWAY, placement_position)
        if self.minerals > 1000 and '14:00' <= self.time_formatted and self.townhalls.amount >= 4:

            if self.structures(UnitTypeId.GATEWAY).amount + self.structures(
                    UnitTypeId.WARPGATE).amount < 20 and self.already_pending(
                    UnitTypeId.GATEWAY) <= 4 and self.can_afford(UnitTypeId.GATEWAY):
                if self.can_afford(UnitTypeId.GATEWAY):
                    building_place = self.structures(UnitTypeId.PYLON).random.position
                    placement_position = await self.find_placement(UnitTypeId.GATEWAY, near=building_place,
                                                                   placement_step=2)
                    worker_candidates = self.workers.filter(lambda worker: (
                                                                                   worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)
                    if placement_position:
                        build_worker = worker_candidates.closest_to(placement_position)
                        build_worker.build(UnitTypeId.GATEWAY, placement_position)

    async def attack(self):

        if self.supply_army > 70 and self.rally_defend == False:
            await self.attack_()
            target = self.select_target()
            for unit in self.units.of_type(
                    {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.VOIDRAY, UnitTypeId.STALKER, UnitTypeId.ADEPT,
                     UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR, UnitTypeId.OBSERVER, UnitTypeId.CHANGELINGZEALOT}):
                self.army_units.append(unit)
                unit.attack(target)

    async def attack_(self):
        attack_units = [UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER, UnitTypeId.SENTRY,
                        UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
                        UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
                        UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
                        UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
                        UnitTypeId.CHANGELINGZEALOT]
        if any(self.units(unit_type).amount > 0 for unit_type in attack_units):
            enemy_start_location_cleared = not self.enemy_units.exists and self.is_visible(
                self.enemy_start_locations[0])

            if enemy_start_location_cleared:
                for unit_type in attack_units:
                    await self.assign_units_to_resource_clusters(unit_type)
            else:
                for unit_type in attack_units:
                    units = self.units(unit_type).idle
                    for unit in units:
                        unit.attack(self.enemy_start_locations[0])

    async def assign_units_to_resource_clusters(self, unit_type):
        units = self.units(unit_type).idle
        resource_clusters = self.expansion_locations_list

        if units.exists and resource_clusters:
            # 为每个单位随机分配一个目标资源点
            for unit in units:
                target = random.choice(resource_clusters)
                unit.attack(target)

        # tank : Unit
        # if self.units(UnitTypeId.SIEGETANK).exists:
        #     tank = self.units(UnitTypeId.SIEGETANK).closest_to(random.choice(self.enemy_start_locations))
        #
        # for unit in self.marine_tag_list:
        #     unit.move(random.choice(self.enemy_start_locations))
        # for unit in self.tank_tag_list:
        #     unit.attack(random.choice(self.enemy_start_locations))
        # for unit in self.medic_tag_list:
        #     unit.move(tank.position)

    # async def back_to_rally(self):
    #     force = self.units.of_type({UnitTypeId.MARINE,UnitTypeId.SIEGETANK,UnitTypeId.MEDIVAC})
    #     if force.exists:
    #         map_center = self.game_info.map_center
    #         rally_point = self.structures.find_by_tag(self.center_tag_list[-1]).position.towards(map_center,distance=5)
    #         for unit in force:
    #             if unit not in self.unit_tags_received_action and unit.distance_to(rally_point) > 10:
    #                 unit.move(rally_point)

    async def defend(self):
        print("Defend:", self.rally_defend)
        print("Attack:", self.attacking)
        if self.townhalls.exists:
            for nexus in self.townhalls:
                if self.enemy_units.amount >= 2 and self.enemy_units.closest_distance_to(nexus) < 30:
                    self.rally_defend = True
                    for unit in self.units.of_type(
                            {UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.VOIDRAY, UnitTypeId.STALKER, UnitTypeId.ADEPT,
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
                         UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR, UnitTypeId.OBSERVER, UnitTypeId.CHANGELINGZEALOT}):
                    if unit.distance_to(self.start_location) > 100 and unit not in self.unit_tags_received_action:
                        unit.move(rally_point)

    def select_target(self) -> Point2:
        if self.enemy_structures:
            return random.choice(self.enemy_structures).position
        return self.enemy_start_locations[0]
    async def research_blink(self):
        if UpgradeId.CHARGE in self.state.upgrades and self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
            vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).random
            abilities = await self.get_available_abilities(vc)
            if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready and self.already_pending(UpgradeId.BLINKTECH) == 0:
                if self.can_afford(UpgradeId.BLINKTECH) and vc.is_idle and AbilityId.RESEARCH_BLINK in abilities:
                    vc.research(UpgradeId.BLINKTECH)
    async def stalker_blink(self):
        if self.units(UnitTypeId.STALKER).exists and UpgradeId.BLINKTECH in self.state.upgrades:

            stalkers = self.units(UnitTypeId.STALKER)
            for stalker in stalkers:
                abilities = await self.get_available_abilities(stalker)
                if (
                        stalker.health_percentage < .5
                        and stalker.shield_health_percentage < .3
                        and AbilityId.EFFECT_BLINK_STALKER in abilities
                ):
                    if self.enemy_units:
                        enemy = self.enemy_units.closest_to(stalker)
                        stalker(AbilityId.EFFECT_BLINK_STALKER, stalker.position.towards(enemy, -6))
    @staticmethod
    def neighbors4(position, distance=1) -> Set[Point2]:
        p = position
        d = distance
        return {Point2((p.x - d, p.y)), Point2((p.x + d, p.y)), Point2((p.x, p.y - d)), Point2((p.x, p.y + d))}

    # Stolen and modified from position.py
    def neighbors8(self, position, distance=1) -> Set[Point2]:
        p = position
        d = distance
        return self.neighbors4(position, distance) | {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d)),
        }


def main():
    run_game(
        maps.get(ladder_map_pool_2022_07[random.randint(0, 6)]),
        [Bot(Race.Protoss, WarpGateBot()), Computer(Race.Zerg, Difficulty.VeryHard)],
        realtime=False,
    )


if __name__ == "__main__":
    main()
