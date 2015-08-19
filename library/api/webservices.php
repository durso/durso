<?php

namespace library\api;

class webservices{
    protected $url;
    protected $qs;
    protected $ch;
    
    public function __construct($url,$qs = ""){
        $this->url = $url;
        $this->qs = $qs;
        $this->ch = curl_init(); 
    }
    
    public function addParam($key,$value){
        $this->qs .= $this->qs == "" ? "?":"&";
        $this->qs .= "$key=$value";
    }
    public function addopt($const, $value){
        curl_setopt($this->ch, $const, $value);
    }

    public function exec(){
        $this->addopt(CURLOPT_URL,$this->url.$this->qs);
        $this->addopt(CURLOPT_RETURNTRANSFER,true);
        return curl_exec($this->ch);
    }
    
    public function close(){
        curl_close($this->ch);
    }
    
}