<?php


/**
 * Description of block
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;

class block extends paired{

    
    public function __construct($tag = "h1") {
        parent::__construct();
        $this->tag = $tag;
    }


}