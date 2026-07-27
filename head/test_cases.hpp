#pragma once

#include <string>
#include <vector>
#include "types.hpp"

enum class BoundaryType { Periodic, Transmissive };
struct BoundaryConfig {
    BoundaryType left=BoundaryType::Transmissive;
    BoundaryType right=BoundaryType::Transmissive;
    BoundaryType bottom=BoundaryType::Transmissive;
    BoundaryType top=BoundaryType::Transmissive;
};
enum class CaseId { KelvinHelmholtz, ShockBubble, BlastWave };
struct CaseConfig {
    int nx,ny,ng;
    double x_min,x_max,y_min,y_max,cfl,t_end,gamma;
    BoundaryConfig bc;
    std::vector<double> snapshot_times={};
    std::vector<std::string> snapshot_tags={};
};

CaseId parse_case_id(const std::string&);
std::string case_id_to_string(CaseId);
CaseConfig get_case_config(CaseId);
CaseConfig get_case_config(const std::string&);
CaseConfig get_n_case_config(CaseId,int);
CaseConfig get_n_case_config(const std::string&,int);
Conserved initial_state_at(CaseId,double,double);
Conserved initial_state_at(const std::string&,double,double);
