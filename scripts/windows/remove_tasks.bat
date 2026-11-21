@echo off
schtasks /Delete /F /TN "NICS Trading self-learning"
schtasks /Delete /F /TN "NICS Trading shadow score"
schtasks /Delete /F /TN "NICS Trading opinion"
