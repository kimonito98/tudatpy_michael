/*    Copyright (c) 2010-2026, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 *
 *    References:
 *          ITU-R Recommendation P.531-15 (2023), Ionospheric propagation data and prediction methods
 *              required for the design of satellite networks and systems.
 *          B. Nava, P. Coisson, and S.M. Radicella (2008), A new version of the NeQuick ionosphere
 *              electron density model, J. Atmos. Sol.-Terr. Phys., 70(15), doi:10.1016/j.jastp.2008.01.015
 */

#ifndef TUDAT_NEQUICK2MODEL_H
#define TUDAT_NEQUICK2MODEL_H

#include <functional>
#include <cmath>
#include <Eigen/Core>

#include "tudat/astro/basic_astro/ionosphereModel.h"
#include "tudat/io/readNeQuick2Data.h"

namespace tudat
{
namespace environment
{

//! NeQuick-2 ionospheric layer parameters at a specific location and time.
struct NeQuick2LayerParameters
{
    double foE = 0.0;     //!< E-layer critical frequency [MHz]
    double foF1 = 0.0;    //!< F1-layer critical frequency [MHz]
    double foF2 = 0.0;    //!< F2-layer critical frequency [MHz]
    double M3000 = 0.0;   //!< M(3000)F2 propagation factor

    double hmE = 120.0;   //!< E-layer peak height [km]
    double hmF1 = 0.0;    //!< F1-layer peak height [km]
    double hmF2 = 0.0;    //!< F2-layer peak height [km]

    double NmE = 0.0;     //!< E-layer peak electron density [10^11 el/m^3]
    double NmF1 = 0.0;    //!< F1-layer peak electron density [10^11 el/m^3]
    double NmF2 = 0.0;    //!< F2-layer peak electron density [10^11 el/m^3]

    //! Epstein layer amplitudes aep[0]=F2, aep[1]=F1, aep[2]=E  [10^11 el/m^3]
    double aep[ 3 ] = { 0.0, 0.0, 0.0 };

    //! Layer thickness parameters bb[0..5]:
    //! bb[0]=BEbot, bb[1]=BEtop, bb[2]=B1bot, bb[3]=B1top, bb[4]=B2bot, bb[5]=B2top (=H0 for topside)
    double bb[ 6 ] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
};

//! NeQuick-2 electron density model (ITU-R P.531).
/*!
 * Computes 3D electron density N_e(h, lat, lon, time) using the NeQuick-2 formulation
 * based on Epstein layers anchored to E, F1, and F2 layer parameters derived from
 * CCIR spherical harmonic coefficients and solar activity input.
 *
 * Reference: ITU-R Recommendation P.531, NeQuick2 P.531-15 release (ICTP/ESA).
 */
class NeQuick2Model : public IonosphereModel
{
public:

    //! Constructor.
    /*!
     * \param ccirData CCIR coefficient data (12 monthly sets of foF2 and M(3000)F2 coefficients).
     * \param modipGrid MODIP grid matrix with boundary padding (184 x 184).
     * \param solarFluxFunction Function returning F10.7 solar radio flux [SFU] as a function of time [s since J2000].
     * \param inputIsEffectiveIonizationLevel If true, the function returns effective ionization level Az instead of F10.7.
     */
    NeQuick2Model(
        const input_output::CcirData& ccirData,
        const Eigen::MatrixXd& modipGrid,
        std::function< double( double ) > solarFluxFunction,
        bool inputIsEffectiveIonizationLevel = false );

    //! Compute electron density at a point [el/m^3].
    /*!
     * \param heightKm Height above surface [km].
     * \param latitudeDeg Geographic latitude [degrees N].
     * \param longitudeDeg Geographic longitude [degrees E].
     * \param month Month (1..12).
     * \param flx F10.7 solar radio flux [SFU].
     * \param ut Universal Time [hours].
     * \return Electron density in el/m^3.
     */
    double computeElectronDensity( double heightKm, double latitudeDeg, double longitudeDeg,
                                   int month, double flx, double ut ) const;

    //! Compute electron density using time in seconds since J2000 (calls internal solar flux function).
    double computeElectronDensityAtTime( double heightKm, double latitudeDeg, double longitudeDeg,
                                         double time ) const;

    //! Compute layer parameters at a given location and time.
    NeQuick2LayerParameters computeLayerParameters( double latitudeDeg, double longitudeDeg,
                                                    int month, double flx, double ut ) const;

    //! Compute electron density from pre-computed layer parameters.
    static double computeElectronDensityFromParams( double heightKm, const NeQuick2LayerParameters& params );

    //! Compute electron density with NmF2 scaled by a factor k.
    //! This is used for IONEX-constrained rescaling.
    double computeElectronDensityRescaled( double heightKm, double latitudeDeg, double longitudeDeg,
                                           int month, double flx, double ut, double nmF2ScaleFactor ) const;

    // --- IonosphereModel interface ---

    //! Get vertical TEC at a location and time [TECU].
    //! Latitude and longitude in degrees, time in seconds since J2000.
    double getVerticalTotalElectronContent( double latitudeDeg, double longitudeDeg, double time ) override;

    //! Get vertical TEC integrated over a finite altitude band [TECU].
    /*!
     * Used by the topside-aware IONEX rescaling: by integrating between two heights one can
     * estimate the column NeQuick attributes to the part of the atmosphere a given link does
     * not traverse (e.g., above the ISS).
     *
     * \param latitudeDeg Geographic latitude [deg N].
     * \param longitudeDeg Geographic longitude [deg E].
     * \param time Time [s since J2000].
     * \param lowAltitudeKm Lower altitude bound of the integral [km].
     * \param highAltitudeKm Upper altitude bound of the integral [km].
     * \return Partial vertical TEC in TECU (zero if highAltitudeKm <= lowAltitudeKm).
     */
    double getVerticalTotalElectronContentInBand(
        double latitudeDeg, double longitudeDeg, double time,
        double lowAltitudeKm, double highAltitudeKm );

    //! Get reference ionosphere height [m]. Returns a nominal hmF2 of 350 km.
    double getReferenceIonosphereHeight( ) const override
    {
        return 350.0e3;
    }

    //! Get the solar flux function.
    std::function< double( double ) > getSolarFluxFunction( ) const { return solarFluxFunction_; }

    //! Convert time [s since J2000] to month and UT [hours].
    static void timeToMonthAndUT( double time, int& month, double& ut );

private:

    //! Compute MODIP from geographic coordinates [degrees].
    double computeModip( double latitudeDeg, double longitudeDeg ) const;

    //! 3rd-order interpolation (Fortran finter3).
    static double finter3( const double z[ 4 ], double x );

    //! Compute foF2 and M(3000)F2 from CCIR coefficients.
    void computeFoF2AndM3000( double modip, int month, double ut, double R12,
                              double latDeg, double lonDeg,
                              double& foF2, double& M3000 ) const;

    //! Spherical harmonic evaluation (Fortran gamma1).
    double gamma1( double modip, double latDeg, double lonDeg, double hour,
                   int iharm, const int* nq, int k1, int m, int mm, int m3,
                   const double* sfe ) const;

    //! Compute foE and foF1 (Fortran ef1r).
    static void computeELayerAndF1( double latDeg, int month, double flx, double chi,
                                    double foF2, double& foE, double& foF1 );

    //! Compute solar declination sine and cosine (Fortran sdec).
    static void solarDeclination( int month, double ut, double& sinDelta, double& cosDelta );

    //! Compute F2-layer peak height [km] (Fortran peakh).
    static double computeHmF2( double foE, double foF2, double M3000 );

    //! Prepare bottomside layer parameters (Fortran prepmdgr).
    static void prepareLayerParameters( int month, double R12, double foF2, double foF1, double foE,
                                        double M3000, NeQuick2LayerParameters& params );

    //! Bottomside electron density from Epstein layers [el/m^3] (Fortran NeMdGR).
    static double bottomsideElectronDensity( const NeQuick2LayerParameters& params, double heightKm );

    //! Topside electron density [el/m^3] (Fortran topq).
    static double topsideElectronDensity( double heightKm, double NmaxAtPeak, double hmaxKm, double H0 );

    //! Safe exponential (Fortran fexp): clamped to avoid overflow/underflow.
    static double safeExp( double a );

    //! Smooth join function (Fortran djoin).
    static double djoin( double f1, double f2, double alpha, double x );

    //! Convert F10.7 solar flux to R12 sunspot number.
    static double fluxToR12( double flx );

    // Data members
    input_output::CcirData ccirData_;
    Eigen::MatrixXd modipGrid_;
    std::function< double( double ) > solarFluxFunction_;
};

//! IONEX-constrained NeQuick-2 wrapper.
/*!
 * Rescales the NeQuick-2 NmF2 so that a vertical TEC anchor matches the IONEX VTEC at the
 * ionospheric pierce point, then integrates the rescaled electron density along the
 * transmitter-receiver ray path to compute slant TEC.
 *
 * Two rescaling modes are supported via the `topsideAwareRescaling` constructor flag:
 *
 *  - **Full-column** (legacy, `topsideAwareRescaling = false`):
 *
 *        k = VTEC_IONEX / VTEC_NeQuick(0 → 20000 km)
 *
 *    Calibrates against the entire vertical column. Appropriate when the receiver sits above
 *    the ionosphere (ground-to-GNSS): in that case the link traverses essentially the full
 *    column and the rescaling is consistent.
 *
 *  - **Topside-aware** (default, `topsideAwareRescaling = true`):
 *
 *        cutoff       = max(altitude_tx, altitude_rx)
 *        VTEC_anchor  = max(VTEC_IONEX - VTEC_NeQuick(cutoff → 20000 km),
 *                            rescalingFloor · VTEC_IONEX)
 *        k            = VTEC_anchor / VTEC_NeQuick(0 → cutoff)
 *
 *    Calibrates against only the portion of the column the ray actually traverses. Important
 *    for receivers embedded in the ionosphere (e.g. ISS at ~420 km) where the upper-topside
 *    and plasmaspheric contribution to the IONEX VTEC is not seen by the link. For ground-to-
 *    GNSS links the cutoff is at ~20000 km and the topside-aware result coincides with the
 *    legacy full-column rescaling to within numerical noise.
 *
 * The `rescalingFloor` parameter (default 0.1) clamps the anchor from below: if NeQuick over-
 * estimates the upper-topside column to the point where `VTEC_IONEX - VTEC_NQ_above_cutoff`
 * would be negative or implausibly small, the anchor is replaced by `rescalingFloor · VTEC_IONEX`
 * so the rescaling remains numerically stable.
 */
class IonexConstrainedNeQuick2Model
{
public:

    //! Constructor.
    /*!
     * \param neQuick2Model The underlying NeQuick-2 model.
     * \param ionexModel The IONEX-based tabulated ionosphere model providing VTEC.
     * \param topsideAwareRescaling If true (default), subtract NeQuick's estimate of the
     *        column above the ray's maximum altitude from VTEC_IONEX before forming the
     *        rescaling factor. If false, use the legacy full-column rescaling.
     * \param rescalingFloor Minimum admissible anchor as a fraction of VTEC_IONEX. Only
     *        used when topsideAwareRescaling is true (default 0.1).
     */
    IonexConstrainedNeQuick2Model(
        std::shared_ptr< NeQuick2Model > neQuick2Model,
        std::shared_ptr< IonosphereModel > ionexModel,
        bool topsideAwareRescaling = true,
        double rescalingFloor = 0.1 );

    //! Compute rescaled slant TEC along a ray [el/m^2].
    /*!
     * \param txPositionEarthFixed Transmitter position in Earth body-fixed frame [m].
     * \param rxPositionEarthFixed Receiver position in Earth body-fixed frame [m].
     * \param time Time [s since J2000].
     * \param earthRadius Earth's mean radius [m].
     * \param quadratureOrder Number of Gauss-Legendre quadrature nodes (default 50).
     * \return Slant TEC in el/m^2.
     */
    double computeRescaledSlantTec(
        const Eigen::Vector3d& txPositionEarthFixed,
        const Eigen::Vector3d& rxPositionEarthFixed,
        double time,
        double earthRadius,
        int quadratureOrder = 50 ) const;

    //! Get the underlying NeQuick-2 model.
    std::shared_ptr< NeQuick2Model > getNeQuick2Model( ) const { return neQuick2Model_; }

    //! Get the underlying IONEX model.
    std::shared_ptr< IonosphereModel > getIonexModel( ) const { return ionexModel_; }

    //! True if topside-aware rescaling is enabled.
    bool usesTopsideAwareRescaling( ) const { return topsideAwareRescaling_; }

    //! Fraction of VTEC_IONEX used as a lower clamp on the anchor.
    double getRescalingFloor( ) const { return rescalingFloor_; }

private:

    std::shared_ptr< NeQuick2Model > neQuick2Model_;
    std::shared_ptr< IonosphereModel > ionexModel_;
    bool topsideAwareRescaling_;
    double rescalingFloor_;
};

}  // namespace environment
}  // namespace tudat

#endif  // TUDAT_NEQUICK2MODEL_H
