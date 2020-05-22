#include "Config.hpp"

#include <iostream>
#include <fstream>
#include <stdint.h>

#include <toml++/toml.h>

namespace tppocr {

void Config::parseFromTOML(const std::string path) {
    toml::table table;

    try {
        table = toml::parse_file(path);
    } catch (const toml::parse_error & error) {
        std::cerr
            << "Error parsing file '" << *error.source().path
            << "':\n" << error.description()
            << "\n  (" << error.source().begin << ")"
            << std::endl;
    }

    realTime = table.get("realtime")->as_boolean()->get();
    processingFPS = table.get("processing-fps")->as_floating_point()->get();
    tessdataPath = table.get("tessdata")->as_string()->get();
    detectorModelPath = table.get("detector-model")->as_string()->get();
    detectorConfidenceThreshold = table.get("detector-confidence-threshold")->as_floating_point()->get();
    detectorNonmaximumSuppressionThreshold = table.get("detector-nonmaximum-suppresion-threshold")->as_floating_point()->get();

    for (const auto & node : *table["region"].as_array()) {
        const auto regionConfig = node.as_table();
        Region region;
        region.name = regionConfig->get("name")->as_string()->get();
        region.x = regionConfig->get("x")->as_integer()->get();
        region.y = regionConfig->get("y")->as_integer()->get();
        region.width = regionConfig->get("width")->as_integer()->get();
        region.height = regionConfig->get("height")->as_integer()->get();

        regions.push_back(region);

        std::cerr << "Configured region '" << region.name << "'" << std::endl;
    }
}

}
