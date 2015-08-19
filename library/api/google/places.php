<?php

namespace library\api\google;
use library\api\webservices;

class places extends webservices{
    private $key = "AIzaSyBqKpK4MfNmwcDrHYvpWHZrApHhwrHpn7Y";
    
    public function __construct($url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json") {
        parent::__construct($url);
    }
    public function setKey($key){
        $this->key = $key;
    }
    public function prepare($latitude,$longitude,$name,$radius = 1500){
        $this->addParam("key",$this->key);
        $this->addParam("location",  floatval($latitude).",".floatval($longitude));
        $this->addParam("radius",$radius);
        $this->addParam("keyword",$name);
    }
    public function exec($prepare = true){
        $result = parent::exec();
        $result = json_decode($result,true);
        if($prepare){
            return $result["results"];
        }
        return $result;
    }

    
}