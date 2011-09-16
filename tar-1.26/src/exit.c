/* This file is part of GNU tar.
   Copyright (C) 2009 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 3, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.  */

#include <system.h>
#include "common.h"

void (*fatal_exit_hook) (void);

void
fatal_exit (void)
{
  if (fatal_exit_hook)
    fatal_exit_hook ();
  error (TAREXIT_FAILURE, 0, _("Error is not recoverable: exiting now"));
  abort ();
}

void
xalloc_die (void)
{
  error (0, 0, "%s", _("memory exhausted"));
  fatal_exit ();
}