#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <ctime>
#include <cstring>
#include <iostream>
#include "cutil.h"

//#define PROFILE_TIMERS

class CPUtimer
{
public:
	uint timer;
	float time;
	int ncalls;
	char* name;

	void init(char* name_in)
	{
		cutCreateTimer(&timer);
		ncalls = 0;
		time = 0;
		name = name_in;
	}

	void start_timer(void)
	{
		cutStartTimer(timer);
		ncalls++;
	}

	void stop_timer(void)
	{
		cutStopTimer(timer);
	}

	float read_timer(void)
	{
		return cutGetTimerValue(timer);
	}

	void update_time(void)
	{
		time += cutGetTimerValue(timer);
	}

	float get_time(void)
	{
		return time;
	}

};

extern CPUtimer* g_timers;

extern int itimer_sort;
extern int itimer_chargeassign;
extern int itimer_chargetomesh;
extern int itimer_padvnc;

extern int get_timer_int(const char* name_in);




























