<?php

/**
 * Description of navbar
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\components\component;

class navbar extends component{
    public function __construct() {
        $this->elements["nav"] = new nav(array("navbar")); 
        $this->elements["container"] = new group(); 
        $this->elements["nav"]->addChild($this->elements["container"]);
        $this->elements["header"] = new group();
        $this->elements["container"]->addChild($this->elements["header"]);
        $this->elements["button"] = new button();
    }
    
}
