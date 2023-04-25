#!/bin/bash
scontrol show job $1
# scontrol show job $1 | grep StartTime