//
// Bluegiga's Bluetooth Smart Demo Application
// Contact: support@bluegiga.com.
//
// This is free software distributed under the terms of the MIT license reproduced below.
//
// Copyright (c) 2012, Bluegiga Technologies
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
// EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
//

#include "stdma.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/time.h>
#include <time.h>    // time()
#include <signal.h>

#include "cmd_def.h"

#include "generated/airframe.h"
#include "subsystems/datalink/datalink.h"
#include "mcu_periph/uart.h"
#include "pprzlink/pprz_transport.h"
#include "mcu_periph/sys_time.h"

struct link_device* dev = &(BLUEGIGA_UART.device);

uint8_t stdma_data[31];
uint8_t stdma_data_len = 0;
bool stdma_initialised = false;

struct pprz_transport stdma_trans;

//#define DEBUG

#ifdef PRINT
#define debug_print(...) printf(__VA_ARGS__)
#else
#define debug_print(...)
#endif

// this needs to be defined for two wire uart not for USB
// TODO generalize over usart/usb
#define PACKET_MODE

// {0x0, 0x00, 0x1e, 0x80, 0x07, 0x00} for BLE112D usb dongle
// {0x40, 0xe0, 0x2d, 0x80, 0x07,0x00} for BLE121LR
uint8_t MAC_ADDR[] = {0x40, 0xe0, 0x2d, 0x80, 0x07,0x00};

// ADVertizement data
uint8_t adv_data[31];
uint8_t adv_data_len;

// STDMA variables
const uint8_t STDMA_ADV_DATA_HEADER_LEN = 6;      // length of header for custom data
const uint8_t STDMA_ADV_HEADER_LEN = 10;          // total length of response data package
const uint8_t STDMA_ADV_MAX_DATA_LEN = 21;        // Max data length
const uint8_t STDMA_SLOTS = 8;
// note: transmit packets per frame is fixed to 1

const uint8_t STDMA_MIN_INTERVAL = 3;           // Min nr of frames before required to reselect
const uint8_t STDMA_MAX_INTERVAL = 7;           // Max nr of frames before required to reselect
const uint8_t STDMA_SELECTION_INTERVAL = 2;     // Number of slots to consider around the nominal increment
const uint32_t STDMA_FRAME_DURATION = 6553;       // frame interval [clock ticks, 32768 = 1s]

// STDMA states, listed in decreasing priority: internally allocated, externally allocated, busy, free
const uint8_t STDMA_STATE_INTER_ALLOC = 3;
const uint8_t STDMA_STATE_EXTER_ALLOC = 2;
const uint8_t STDMA_STATE_BUSY = 1;
const uint8_t STDMA_STATE_FREE = 0;

uint8_t stdma_slot_status[8];               // status of each slot stored here
uint8_t stdma_slot_timeout[8];              // time a reservation is valid
uint8_t stdma_next_slot_timeout[8];         // next frame reservation
uint8_t stdma_free_slots[8];                // temporary store of all free slots

uint8_t stdma_my_slot;
uint8_t stdma_my_next_slot;
uint8_t stdma_current_slot;
uint8_t stdma_braodcasting;

// advertisement buffer positions
const uint8_t POS_ADV_LEN = 3;                        // length of advertisement data
const uint8_t POS_ADV_STDMA_OFFSET = 7;               // stdma offset
const uint8_t POS_ADV_STDMA_TIMEOUT = 8;              // stdma timeout
const uint8_t POS_ADV_TX_STRENGTH = 9;                // transmit power dBm

// PPRZ MSG positions
const uint8_t PPRZ_POS_STX = 0;
const uint8_t PPRZ_POS_LEN = 1;
const uint8_t PPRZ_POS_SENDER_ID = 2;
const uint8_t PPRZ_POS_MSG_ID = 3;

static int cmp_addr(const uint8_t first[], const uint8_t second[])
{
  int i;
  for (i = 5; i >= 0; i--) {
    if (first[i] != second[i]) { return 5 - i; }
  }
  return 6;
}

#ifdef DEBUG
static int cmp_bdaddr(const bd_addr first, const bd_addr second)
{
  for (uint8_t i = 0; i < sizeof(bd_addr); i++) {
    if (first.addr[i] != second.addr[i]) { return 1; }
  }
  return 0;
}

static void print_bdaddr(const bd_addr __attribute__((unused)) bdaddr)
{
  debug_print("%02x:%02x:%02x:%02x:%02x:%02x",
              bdaddr.addr[5],
              bdaddr.addr[4],
              bdaddr.addr[3],
              bdaddr.addr[2],
              bdaddr.addr[1],
              bdaddr.addr[0]);
}

static void print_raw_packet(struct ble_header *hdr, uint8_t __attribute__((unused)) *data)
{
  debug_print("Incoming packet (len: %d): ", hdr->lolen);
  for (uint8_t i = 0; i < sizeof(*hdr); i++) {
    debug_print("%02x ", ((unsigned char *)hdr)[i]);
  }
  for (uint8_t i = 0; i < hdr->lolen; i++) {
    debug_print("%02x ", data[i]);
  }
  debug_print("\n");
}
#endif

/* This is a helper function to output data from the BLE-API commands */
#define OUTPUT_BUFFER_SIZE 256
uint8_t output_buffer[OUTPUT_BUFFER_SIZE];
uint16_t write_idx = 0;
uint16_t read_idx = 0;
static void output(uint8_t len1, uint8_t *data1, uint16_t len2, uint8_t *data2)
{
  if ((read_idx - write_idx - 1 + OUTPUT_BUFFER_SIZE) % OUTPUT_BUFFER_SIZE >= len1 + len2){
    memcpy(output_buffer + write_idx, data1, len1);
    write_idx = (write_idx + len1) % OUTPUT_BUFFER_SIZE;
    memcpy(output_buffer + write_idx, data2, len2);
    write_idx = (write_idx + len2) % OUTPUT_BUFFER_SIZE;
  } else {
    // It is possible that something went wrong in reading or writing causing a lock up in the coms
    // this can be identified if the buffer is full
    // If this happens, reset the coms
    //stdma_initialised = false;
    write_idx = read_idx = 0;
    ready_to_send = 1;
    //ble_cmd_system_reset(0);
  }
}

void write_message(void){
  if (ready_to_send && (write_idx - read_idx + OUTPUT_BUFFER_SIZE) % OUTPUT_BUFFER_SIZE > 0){
    uint8_t msg_len = output_buffer[(read_idx+1) % OUTPUT_BUFFER_SIZE] + 4; // message length is header + data (which is indicated by header[1]
    if(dev->check_free_space(dev->periph, 0, msg_len)){
  #ifdef PACKET_MODE
      // prefix total message length byte when in packet mode
      dev->put_byte(dev->periph, 0, msg_len);
  #endif
      for (uint8_t i = 0; i < msg_len; i++){
        dev->put_byte(dev->periph, 0, output_buffer[read_idx]);
        read_idx = (read_idx + 1) % OUTPUT_BUFFER_SIZE;
      }
      ready_to_send = 0;
    }
  }
}

void read_message(void)
{
  static struct ble_header hdr = {0};
  static bool hdr_found = false;
  static uint8_t data_in[256];
  static uint8_t data_index = 0;

  while (dev->char_available(dev->periph)){
    if(!hdr_found && dev->char_available(dev->periph) >= 4){
        hdr.type_hilen = dev->get_byte(dev->periph);
        hdr.lolen = dev->get_byte(dev->periph);
        hdr.cls = dev->get_byte(dev->periph);
        hdr.command = dev->get_byte(dev->periph);
        hdr_found = true;
    }

    if(hdr_found && dev->char_available(dev->periph) >= hdr.lolen){
      for(uint8_t i = 0; i < hdr.lolen; i++){
        data_in[data_index++] = dev->get_byte(dev->periph);
      }

      const struct ble_msg *msg = ble_get_msg_hdr(hdr);

#ifdef DEBUG_RAW
      print_raw_packet(&hdr, data_in);
#endif
      if (!msg) {
        debug_print("ERROR: Unknown message received\n");

      }

      // handle message
      msg->handler(data_in);

      // reset buffer index counters for next message
      hdr_found = false;
      hdr.lolen = 0;
      data_index = 0;
    } else {
      break;
    }
  }
}

void ble_evt_gap_scan_response(const struct ble_msg_gap_scan_response_evt_t *msg)
{
  // check if sender is likely a bluegiga module or dongle
  if (cmp_addr(msg->sender.addr, MAC_ADDR) < 3) { return; }

  // store stdma slot, offset and timeout
  // response data is [header, offset, timeout, data]
  uint8_t temp = stdma_current_slot + msg->data.data[POS_ADV_STDMA_OFFSET];       // next reserved slot
  while (temp > STDMA_SLOTS) { // wrap index on number of slots
    temp = temp - STDMA_SLOTS;
  }

  if (temp != stdma_my_next_slot && msg->data.data[POS_ADV_STDMA_TIMEOUT] > stdma_next_slot_timeout[temp]) {
    stdma_next_slot_timeout[temp]  = msg->data.data[POS_ADV_STDMA_TIMEOUT];
  }

  uint8_t data[128];

  memcpy(data, msg->data.data + STDMA_ADV_HEADER_LEN, msg->data.len - STDMA_ADV_HEADER_LEN);

  // msg form {header, length, ac_id, msg_id, [msg], crc_a, crc_b}
  // rssi message is [rssi, trans_strength}
  uint8_t rssi_msg[8] = {0x99, 8, data[PPRZ_POS_SENDER_ID], 28, msg->rssi, msg->data.data[POS_ADV_TX_STRENGTH], 0, 0};
  
  uint8_t j = 0, ck_a = 0, ck_b = 0;
  for(j = 1; j < 6; j++){
    ck_a += rssi_msg[j];
    ck_b += ck_a;
  }
  
  rssi_msg[6] = ck_a;
  rssi_msg[7] = ck_b;

  memcpy(data + msg->data.len - STDMA_ADV_HEADER_LEN, rssi_msg, 8);

#ifdef DEBUG
  debug_print("recv'd data: ");
  for (int8_t i = 0; i < msg->data.len - STDMA_ADV_HEADER_LEN + 8; i++) {
    debug_print("%02x ", data[i]);
  }
  debug_print("\n");
#endif

  // parse data
  for (uint8_t i = 0; i < msg->data.len - STDMA_ADV_HEADER_LEN + 8; i++){
    parse_pprz(&stdma_trans, data[i]);
  }
  if (stdma_trans.trans_rx.msg_received){
    dl_msg_available = true;
    DlCheckAndParse(&DOWNLINK_DEVICE.device, &stdma_trans.trans_tx, stdma_trans.trans_rx.payload);
    stdma_trans.trans_rx.msg_received = false;
  }
}

void ble_evt_connection_status(const struct ble_msg_connection_status_evt_t __attribute__((unused)) *msg) {}
void ble_evt_connection_disconnected(const struct ble_msg_connection_disconnected_evt_t __attribute__((unused)) *msg) {}
void ble_evt_attclient_procedure_completed(const struct ble_msg_attclient_procedure_completed_evt_t __attribute__((unused)) *msg) {}
void ble_evt_attclient_group_found(const struct ble_msg_attclient_group_found_evt_t __attribute__((unused)) *msg) {}
void ble_evt_attclient_find_information_found(const struct ble_msg_attclient_find_information_found_evt_t __attribute__((unused)) *msg) {}
void ble_evt_attclient_attribute_value(const struct ble_msg_attclient_attribute_value_evt_t __attribute__((unused)) *msg) {}

void ble_evt_system_protocol_error(const struct ble_msg_system_protocol_error_evt_t __attribute__((unused)) *msg)
{
  ready_to_send = 1;
}

/* set_stdma_data - set output
 *
 */
void set_stdma_data(uint8_t *data, uint8_t data_len)
{
  stdma_data_len = data_len;
  memcpy(stdma_data, data, stdma_data_len);
}

/* paparazzi specific initialisation */
void stdma_init(void){
  pprz_transport_init(&stdma_trans);
  uart_periph_init(dev->periph);

  // set output function
  bglib_output = output;

  // Reset dongle to get it into known state
  ble_cmd_system_reset(0);
}

static void stdma_start(void)
{
  // initialise STDMA parameters
  memset(stdma_slot_status, STDMA_STATE_FREE, STDMA_SLOTS);
  memset(stdma_slot_timeout, 0, STDMA_SLOTS);
  memset(stdma_next_slot_timeout, 0, STDMA_SLOTS);
  memset(stdma_free_slots, 0, STDMA_SLOTS);

  memset(stdma_data, 0, 31);
  stdma_data_len = 0;

  memset(adv_data, 0, STDMA_ADV_MAX_DATA_LEN);
  adv_data_len = 0;

  stdma_my_slot = 0;
  stdma_my_next_slot = 0;
  stdma_current_slot = 0;
  stdma_braodcasting = 0;

  // Initialize ADV data
  // Flags = LE General Discovery, single mode device (02 01 06) flags for discoverable/connectable
  adv_data[0] = 0x02;               // ad field length = 2 bytes
  adv_data[1] = gap_ad_type_flags;  // ad field type = 0x01 (Flags)
  adv_data[2] = 0x06;               // flags = 0x06, bit 1 General discoverable, bit 2 BR/EDR not supported

  // custom manufacturer
  adv_data[3] = STDMA_ADV_DATA_HEADER_LEN;   // ad field length, minimum 3
  adv_data[4] = 0xff;               // ad field type = 0xFF (Manufacturer Specific Data)
  adv_data[5] = 0xff;               // unknown/prototype Company Identifier Code - octet 2
  adv_data[6] = 0xff;               // unknown/prototype Company Identifier Code - octet 1

  // stdma header
  adv_data[POS_ADV_STDMA_OFFSET] = 0x0;                  // stdma offset
  adv_data[POS_ADV_STDMA_TIMEOUT] = 0x0;                 // stdma timeout

  // TX power in dBm for BLE121LR USB Dongle
  adv_data[POS_ADV_TX_STRENGTH] = 11;

  // set advertisement interval on all three spi_channels
  // increments of 625us
  // range (0x20 - 0x4000)
  // 0x07: All three channels are used
  // 0x03: Advertisement channels 37 and 38 are used.
  // 0x04: Only advertisement channel 39 is used
  ble_cmd_gap_set_adv_parameters(0x20, 0x28, 0x07);

  // set scan parameters interval/window/use active scanning
  // the scan interval defines the period between restarting a scan, each new scan will switch to a new channel
  // increments of 625us
  // range: 0x4 - 0x4000
  // with active scanning receiver will send a scan response msg
  ble_cmd_gap_set_scan_parameters(0x20, 0x20, 0);     // the values selected should be a multiple of the stdma interval

  // set name of device
  char name[256] = {"Bluegiga    "};
  name[11] = '0' + (AC_ID % 10);
  name[10] = '0' + ((AC_ID/10) % 10);
  name[9] = '0' + ((AC_ID/100) % 10);

  // todo add AC_ID here
  ble_cmd_attributes_write(3, 0, 12, name);

  /* Intialize random number generator */
  srand(sys_time.nb_tick);

  ble_cmd_gap_set_mode(gap_general_discoverable, gap_undirected_connectable);

  stdma_initialised = true;
}

/*
 * Wait for Bluegiga module to tell us it is ready to receive messages
 */
void ble_evt_system_boot(const struct ble_msg_system_boot_evt_t __attribute__((unused)) *msg)
{
  ready_to_send = 1;
  // start stdma after receiving boot signal from dongle
  stdma_start();
}

/* stdma_periodic() - staged new advertise message to be set as advertisement
 *
 */
void stdma_periodic(void){
  if (!stdma_initialised) {return;}

  if (stdma_data_len > 0 && stdma_data_len < STDMA_ADV_MAX_DATA_LEN) {
    adv_data_len = stdma_data_len;
    memcpy(adv_data + STDMA_ADV_HEADER_LEN, stdma_data, stdma_data_len);
    stdma_data_len = 0;
  }

  static uint8_t skip = 0;
  int8_t i = 0;
  int8_t j = 0;
  int8_t k = 0;

  // stop broadcasting if I just was
  if (stdma_braodcasting == 1) {
    ble_cmd_gap_set_mode(0, 0); // stop advertisement
    ble_cmd_gap_discover(gap_discover_observation);     // scan for other modules to get rssi values
    stdma_braodcasting = 0;
  }

  if (++stdma_current_slot == STDMA_SLOTS) {       // end of frame
    stdma_current_slot = 0;                        // wrap slot counter

    // decrement timeout values
    i = 0;
    while (i < STDMA_SLOTS) {
      if (stdma_next_slot_timeout[i] > stdma_slot_timeout[i]) { // copy next statuses to list
        stdma_slot_timeout[i] = stdma_next_slot_timeout[i];
        stdma_slot_status[i] = STDMA_STATE_EXTER_ALLOC;
      }
      stdma_next_slot_timeout[i] = 0;

      if (stdma_slot_timeout[i] > 0) {
        stdma_slot_timeout[i] = stdma_slot_timeout[i] - 1;
      }
      if (stdma_slot_timeout[i] == 0) {
        stdma_slot_status[i] = STDMA_STATE_FREE;  // update slot statuses
      }
      i++;
    }

    // check if my slot is about to expire, if so then broadcast and select new one
    // This logic give the same location double the possible slot options
    if (stdma_slot_timeout[stdma_my_slot] == 0) {
      // find free slots in selection interval
      k = 1;
      stdma_free_slots[0] = stdma_my_slot;
      i = 0 - STDMA_SELECTION_INTERVAL;
      while (i <= STDMA_SELECTION_INTERVAL) {
        j = stdma_my_slot + i;
        // bound in [0,STDMA_SLOTS)
        while (j < 0) {
          j = j + STDMA_SLOTS;
        }
        while (j >= STDMA_SLOTS) {
          j = j - STDMA_SLOTS;
        }

        if (stdma_slot_status[j] == STDMA_STATE_FREE) {
          stdma_free_slots[k] = j;
          k = k + 1;
        }
        i = i + 1;
      }

      // determine next slot using random offset
      stdma_my_next_slot = stdma_free_slots[rand() % k];
      stdma_slot_status[stdma_my_next_slot] = STDMA_STATE_INTER_ALLOC;

      // determine new timeout
      stdma_slot_timeout[stdma_my_next_slot] = rand() % (STDMA_MAX_INTERVAL - STDMA_MIN_INTERVAL) + STDMA_MIN_INTERVAL;
    }
  }

  if (stdma_current_slot == stdma_my_slot) {
    if (skip == 1){
      skip = 0;
    } else {
      stdma_my_slot = stdma_my_next_slot;

      // set timeout
      adv_data[POS_ADV_STDMA_TIMEOUT] = stdma_slot_timeout[stdma_my_slot];

      // set advertisement data
      adv_data[POS_ADV_STDMA_OFFSET] = stdma_my_slot - stdma_current_slot + STDMA_SLOTS;    // offset to next transmission

      if (adv_data[POS_ADV_STDMA_OFFSET] < STDMA_SLOTS){
        adv_data[POS_ADV_STDMA_OFFSET] += STDMA_SLOTS;
        skip = 1;
        stdma_slot_timeout[stdma_my_slot] += 1;
      }

      adv_data_len = STDMA_SLOTS;
      memcpy(adv_data + STDMA_ADV_HEADER_LEN, stdma_slot_timeout, STDMA_SLOTS);

      adv_data[POS_ADV_LEN] = adv_data_len + STDMA_ADV_DATA_HEADER_LEN;
      ble_cmd_gap_set_adv_data(0, STDMA_ADV_HEADER_LEN + adv_data_len, adv_data);          // Set advertisement data

  #ifdef DEBUG
      int g;
      debug_print("advertise data: ");
      for (g = 0; g < STDMA_ADV_HEADER_LEN + adv_data_len; g++) {
        debug_print("%02x ", adv_data[g]);
      }
      debug_print("\n");
  #endif

      // broadcast!
      ble_cmd_gap_end_procedure();                        // disable scan

      // enable advertisement
      stdma_braodcasting = 1;
      ble_cmd_gap_set_mode(gap_user_data, gap_scannable_non_connectable);
    }
  }
#ifdef DEBUG
  int s;
  debug_print("timeout: ");
  for (s = 0; s < STDMA_SLOTS; s++) {
    debug_print("%02x ", stdma_slot_timeout[s]);
  }
  debug_print("\n");
#endif
}
