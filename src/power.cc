/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, romain.pereira@inria.fr + rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

# include <xkrt/runtime.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/logger/metric.h>

XKRT_NAMESPACE_BEGIN;

void
runtime_t::power_start(
    const device_unique_id_t device_unique_id,
    power_t * power
) {
    assert(power);

    device_t * device = this->device_get(device_unique_id);
    driver_t * driver = this->driver_get(device->driver_type);
    if (driver->f_power_start)
        driver->f_power_start(device->driver_id, power);
}

void
runtime_t::power_stop(
    const device_unique_id_t device_unique_id,
    power_t * power
) {
    assert(power);

    device_t * device = this->device_get(device_unique_id);
    driver_t * driver = this->driver_get(device->driver_type);
    if (driver->f_power_stop)
        driver->f_power_stop(device->driver_id, power);
    else
    {
        power->dt = -1.0;
        power->P  = -1.0;
    }
}

XKRT_NAMESPACE_END;
